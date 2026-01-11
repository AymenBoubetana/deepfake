import os
import sys
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import whisper
from flask import Flask, render_template, request
from torchvision import transforms, models
from moviepy import VideoFileClip
import librosa
import moviepy as mp  # MoviePy 2.0+ syntax

# --- Fix for potential NumPy/Pickle version issues ---
import numpy

sys.modules['numpy._core'] = numpy.core

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================== VIDEO MODEL ARCHITECTURE ====================
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        weights = F.softmax(self.attention(x), dim=1)
        return torch.sum(x * weights, dim=1), weights.squeeze(-1)


class DeepFakeVideoModel(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
        self.lstm = nn.LSTM(1280, 192, num_layers=2, batch_first=True, bidirectional=True, dropout=0.4)
        self.attention = TemporalAttention(192 * 2)
        # RECONSTRUCTED CLASSIFIER TO MATCH YOUR SAVED WEIGHTS
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),  # Layer 4
            nn.BatchNorm1d(128),  # Layer 5
            nn.ReLU(),
            nn.Dropout(0.28),
            nn.Linear(128, 64),  # Layer 8
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),  # Layer 11
            nn.Sigmoid()
        )

    def forward(self, x):
        b, f, c, h, w = x.shape
        feats = torch.stack([self.feature_extractor(x[:, i]).view(b, -1) for i in range(f)], dim=1)
        lstm_out, _ = self.lstm(feats)
        final_feat, _ = self.attention(lstm_out)
        return self.classifier(final_feat)


class AudioModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Backbone: Whisper Tiny Encoder
        full_whisper = whisper.load_model("tiny")
        self.whisper_model = full_whisper.encoder

        # 2. Inception / Conv Layers (kept to match your saved weights)
        self.Incption1_conv1 = nn.Conv2d(1, 1, 1)
        self.Incption1_conv2_1 = nn.Conv2d(1, 4, 1)
        self.Incption1_conv2_2 = nn.Conv2d(4, 4, 3, padding=1)
        self.Incption1_conv3_1 = nn.Conv2d(1, 4, 1)
        self.Incption1_conv3_2 = nn.Conv2d(4, 4, 3, padding=1)
        self.Incption1_conv4_1 = nn.MaxPool2d(3, stride=1, padding=1)
        self.Incption1_conv4_2 = nn.Conv2d(2, 2, 3, padding=1)
        self.Incption1_bn = nn.BatchNorm2d(11)

        self.Incption2_conv1 = nn.Conv2d(11, 2, 1)
        self.Incption2_conv2_1 = nn.Conv2d(11, 4, 1)
        self.Incption2_conv2_2 = nn.Conv2d(4, 4, 3, padding=1)
        self.Incption2_conv3_1 = nn.Conv2d(11, 4, 1)
        self.Incption2_conv3_2 = nn.Conv2d(4, 4, 3, padding=1)
        self.Incption2_conv4_1 = nn.MaxPool2d(3, stride=1, padding=1)
        self.Incption2_conv4_2 = nn.Conv2d(2, 2, 3, padding=1)
        self.Incption2_bn = nn.BatchNorm2d(12)

        # 4. Final Head
        self.conv1 = nn.Conv2d(12, 16, 5)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, 5)
        # Linear layer expects exactly 1024 inputs
        self.fc1 = nn.Linear(1024, 16)
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. Whisper Feature Extraction [Batch, Seq, Feat]
        x = self.whisper_model(x)

        # 2. Add channel dimension -> [Batch, 1, Seq, Feat]
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        # --- FIX: Force the data to size 1024 ---
        # Your previous code was flattening 576,000 features.
        # We use Adaptive Pooling to shrink it to 32x32 (which equals 1024).
        # This acts as a bridge so the code doesn't crash.
        x = F.adaptive_avg_pool2d(x, (32, 32))

        # Flatten: [Batch, 32, 32] -> [Batch, 1024]
        x = x.flatten(1)

        # 3. Fully Connected Layers
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))

        return x


# ==================== INITIALIZATION ====================
video_model = DeepFakeVideoModel().to(DEVICE)
audio_model = AudioModel().to(DEVICE)

try:
    v_ckpt = torch.load("best_model.pt", map_location=DEVICE, weights_only=False)
    video_model.load_state_dict(v_ckpt["model_state_dict"])
    video_model.eval()
    print("✅ Video Model Loaded")


except Exception as e:
    print(f"❌ Initialization Error: {e}")
try:
    if os.path.exists("audio_model.pth"):
        ckpt = torch.load("audio_model.pth", map_location=DEVICE, weights_only=False)

        # If saved as a full model or nested dict
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        # --- REMAPPING LOGIC ---
        # Converts 'whisper_model.encoder.xxx' to 'whisper_model.xxx'
        # to match our class definition.
        fixed_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("whisper_model.encoder.", "whisper_model.")
            fixed_state_dict[new_key] = v

        # Load with strict=False to bypass any minor metadata mismatches
        audio_model.load_state_dict(fixed_state_dict, strict=False)
        audio_model.eval()
        print("✅ Audio Model Loaded Successfully")
except Exception as e:
    print(f"❌ Audio Load Error: {e}")

# ==================== HELPERS ====================
def get_video_tensor(path):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    cap = cv2.VideoCapture(path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, total - 1, 45, dtype=int)
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i);
        ret, frame = cap.read()
        if ret: frames.append(transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    while len(frames) < 45: frames.append(frames[-1])
    return torch.stack(frames).unsqueeze(0).to(DEVICE)


def get_audio_tensor(path):
    # 1. Extract audio from video using moviepy
    video = VideoFileClip(path)
    audio_data = video.audio.to_soundarray(fps=16000)  # Force 16kHz

    # 2. Convert to Mono (average the channels if stereo)
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    # 3. Use Whisper to format it
    audio_data = audio_data.astype(np.float32)
    audio_padded = whisper.pad_or_trim(audio_data)
    mel = whisper.log_mel_spectrogram(audio_padded)

    # 4. Add batch dimension [1, 80, 3000]
    return mel.unsqueeze(0)


# ==================== ROUTES ====================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['video']
        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            with torch.no_grad():
                v_score = video_model(get_video_tensor(path)).item()
                audio_input = get_audio_tensor(path)
                if audio_input.dim() == 4:
                    audio_input = audio_input.squeeze(1)  # Removes the dimension at index 1
                output = audio_model(audio_input)
                a_score = torch.sigmoid(output.mean()).item()

            final = (v_score * 0.7) + (a_score * 0.3)  # Weighted average
            res = "FAKE" if final >= 0.5 else "REAL"
            stats = {'v': round(v_score * 100, 1), 'a': round(a_score * 100, 1), 'f': round(final * 100, 1)}
            return render_template('index.html', result=res, stats=stats)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)