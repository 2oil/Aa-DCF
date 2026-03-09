#!/usr/bin/env python3

import os, sys, yaml, torch
import soundfile as sf
import numpy as np
from tqdm import tqdm

# ================================
# Paths
# ================================
PROTOCOL = "/home/eoil/Interspeech2026/PA/ASVspoof2019_PA_asv_protocols/ASVspoof2019.PA.asv.eval.gi.trl.txt"
AUDIO_DIR = "/home/eoil/Interspeech2026/PA/ASVspoof2019_PA_eval/flac"
OUT_TXT = "/home/eoil/Interspeech2026/Pa-DCF/Score/pa_aasist_PA19_27.txt"

MODEL_PATH = "/home/eoil/Interspeech2026/aasist/exp_result/PA_AASIST_ep100_bs24/weights/epoch_27_8.052.pth"
CONFIG_PATH = "/home/eoil/Interspeech2026/aasist/exp_result/PA_AASIST_ep100_bs24/config.conf"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.aasist import Model


# ================================
# Load model
# ================================
config = yaml.safe_load(open(CONFIG_PATH))
model = Model(d_args=config["model_config"], device=DEVICE).to(DEVICE)

state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)

model.eval()


# ================================
# Audio loader
# ================================
def load_audio(path, length=64600):

    wav, sr = sf.read(path, dtype="float32")

    if wav.ndim == 2:
        wav = wav.mean(1)

    if len(wav) < length:
        wav = np.tile(wav, length // len(wav) + 1)

    wav = wav[:length]

    return torch.tensor(wav).unsqueeze(0).to(DEVICE)


# ================================
# Read protocol
# ================================
trials = []

with open(PROTOCOL) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 5:
            trials.append(parts[:5])

print("Trials:", len(trials))


# ================================
# Scoring
# ================================
os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)

with open(OUT_TXT, "w") as fw:

    fw.write("utt_id,enroll_id,label,trial_type,cm_score\n")

    for enroll, utt, _, label, trial_type in tqdm(trials):

        path = os.path.join(AUDIO_DIR, f"{utt}.flac")

        if not os.path.exists(path):
            continue

        wav = load_audio(path)

        with torch.no_grad():
            out = model(wav)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            score = logits[0,1].item()

        fw.write(f"{utt},{enroll},{label},{trial_type},{score:.6f}\n")

print("Saved:", OUT_TXT)