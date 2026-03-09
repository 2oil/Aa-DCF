#!/usr/bin/env python3

import os
import sys
import yaml
import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm

# ======================================================
# Paths
# ======================================================
PROTOCOL_FILE = "/home/eoil/AGENT/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt"
AUDIO_DIR = "/home/eoil/AGENT/LA/ASVspoof2019_LA_eval/flac"
OUTPUT_TXT = "/home/eoil/Interspeech2026/a-EER/Score/aasist_LA19.txt"

MODEL_PATH = "/home/eoil/aasist/models/weights/AASIST.pth"
CONFIG_PATH = "/home/eoil/AGENT/AASIST_conf.yaml"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# Import model
# ======================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.aasist import Model

# ======================================================
# Load model
# ======================================================
config = yaml.safe_load(open(CONFIG_PATH))
model = Model(d_args=config["model_config"], device=DEVICE).to(DEVICE)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)

model = torch.nn.DataParallel(model).to(DEVICE)
model.eval()

# ======================================================
# Audio loader
# ======================================================
def load_audio(path, target_len=64600, target_sr=16000):

    wav, sr = sf.read(path, dtype="float32")

    if sr != target_sr:
        raise ValueError(f"Unexpected sample rate: {sr}")

    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    if len(wav) < target_len:
        wav = np.tile(wav, target_len // len(wav) + 1)

    wav = wav[:target_len]

    return torch.tensor(wav).unsqueeze(0).to(DEVICE)

# ======================================================
# Load protocol
# ======================================================
trials = []

with open(PROTOCOL_FILE) as f:

    for line in f:

        parts = line.strip().split()

        if len(parts) < 4:
            continue

        enroll, utt, label, trial_type = parts[:4]

        trials.append((enroll, utt, label, trial_type))

print("Protocol entries:", len(trials))

# ======================================================
# Scoring
# ======================================================
os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)

skipped = []

with open(OUTPUT_TXT, "w") as fw:

    fw.write("utt_id,enroll_id,label,trial_type,cm_score\n")

    for enroll, utt, label, trial_type in tqdm(trials, desc="AASIST scoring"):

        path = os.path.join(AUDIO_DIR, f"{utt}.flac")

        if not os.path.exists(path):
            skipped.append((utt, "file not found"))
            continue

        try:

            wav = load_audio(path)

            with torch.no_grad():
                out = model(wav)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                score = logits[0, 1].item()

            fw.write(f"{utt},{enroll},{label},{trial_type},{score:.6f}\n")

        except Exception as e:

            skipped.append((path, str(e)))

print("Saved scores:", OUTPUT_TXT)

if skipped:

    print("Skipped files:", len(skipped))

    for p, msg in skipped[:5]:
        print(p, "::", msg)