#!/usr/bin/env python3
import os,sys,torch,numpy as np,soundfile as sf
from tqdm import tqdm

PROTOCOL_FILE="/home/eoil/Interspeech2026/PA/ASVspoof2019_PA_asv_protocols/ASVspoof2019.PA.asv.eval.gi.trl.txt"
ENROLL_LIST_FILES=["/home/eoil/Interspeech2026/PA/ASVspoof2019_PA_asv_protocols/ASVspoof2019.PA.asv.eval.female.trn.txt","/home/eoil/Interspeech2026/PA/ASVspoof2019_PA_asv_protocols/ASVspoof2019.PA.asv.eval.male.trn.txt"]
AUDIO_DIR="/home/eoil/Interspeech2026/PA/ASVspoof2019_PA_eval/flac"
OUTPUT_TXT="/home/eoil/Interspeech2026/Pa-DCF/Score/resnet34_pa19.txt"

sys.path.append("/home/eoil/AGENT/")
from ResNetModels.ResNetSE34V2 import MainModel

MODEL_PATH="./ResNetModels/baseline_v2_ap.model"
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

model=MainModel().to(DEVICE)
checkpoint=torch.load(MODEL_PATH,map_location=DEVICE)
state_dict=checkpoint.get("model",checkpoint)
clean_state={}
for k,v in state_dict.items():
    if k.startswith("__S__."): k=k.replace("__S__.","")
    elif k.startswith("__L__."): continue
    clean_state[k]=v
model.load_state_dict(clean_state,strict=True);model.eval()

def read_audio(p):
    wav,_=sf.read(p,dtype="float32")
    if wav.ndim==2: wav=wav.mean(1)
    return wav

@torch.no_grad()
def emb(p):
    w=torch.from_numpy(read_audio(p)).float().unsqueeze(0).to(DEVICE)
    return model(w).squeeze().cpu().numpy()

def cos(a,b):
    d=np.linalg.norm(a)*np.linalg.norm(b)
    return 0.0 if d==0 else float(np.dot(a,b)/d)

enroll_map={}
for path in ENROLL_LIST_FILES:
    with open(path) as f:
        for line in f:
            if not line.strip(): continue
            spk,utts=line.strip().split()[:2]
            enroll_map.setdefault(spk,[]).extend(utts.split(","))
print("Enroll speakers:",len(enroll_map))

enroll_cache={}
def get_enroll(spk):
    if spk in enroll_cache: return enroll_cache[spk]
    embs=[]
    for u in enroll_map.get(spk,[]):
        p=os.path.join(AUDIO_DIR,f"{u}.flac")
        if os.path.exists(p): embs.append(emb(p))
    if not embs: return None
    enroll_cache[spk]=np.mean(embs,0);return enroll_cache[spk]

protocol=[]
with open(PROTOCOL_FILE) as f:
    for line in f:
        p=line.strip().split()
        if len(p)==5: protocol.append((p[0],p[1],p[3],p[4]))
print("Eval trials:",len(protocol))

eval_cache={};results=[];skipped=[]
for spk,utt,label,trial in tqdm(protocol,desc="Scoring"):
    p=os.path.join(AUDIO_DIR,f"{utt}.flac")
    if not os.path.exists(p): skipped.append((utt,"audio missing"));continue
    enroll=get_enroll(spk)
    if enroll is None: skipped.append((utt,"no enroll"));continue
    try:
        if p not in eval_cache: eval_cache[p]=emb(p)
        s=cos(eval_cache[p],enroll)
        results.append((utt,spk,s,label,trial))
    except Exception as e:
        skipped.append((utt,str(e)))

os.makedirs(os.path.dirname(OUTPUT_TXT),exist_ok=True)
with open(OUTPUT_TXT,"w") as f:
    f.write("utt_id,enroll_id,cosine_score,label,trial_type\n")
    for u,e,s,l,t in results: f.write(f"{u},{e},{s:.6f},{l},{t}\n")

print("Saved:",len(results),"->",OUTPUT_TXT)
if skipped:
    print("Skipped:",len(skipped))
    for u,m in skipped[:5]: print(u,"::",m)