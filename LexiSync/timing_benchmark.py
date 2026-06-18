"""
timing_benchmark.py
====================================================================

Παράγει: timing_results.json
"""

import os, re, gc, time, json, sys, subprocess, tempfile, warnings
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings("ignore")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_DIR = "./data"
N_RUNS   = 3
N_PAIRS  = 300
BATCH    = 16

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "thenlper/gte-small",
    "intfloat/e5-small-v2",
    "BAAI/bge-small-en-v1.5",
    "Snowflake/snowflake-arctic-embed-xs",
    "microsoft/deberta-v3-base",
]

DATASETS = {
    "DBLP-Scholar":  {
        "path": f"{BASE_DIR}/DBLP-Scholar",
        "src_cols": ["title","authors","venue","year"],
        "tgt_cols": ["title","authors","venue","year"],
    },
    "Amazon-Google": {
        "path": f"{BASE_DIR}/Amazon-Google",
        "src_cols": ["title","manufacturer","price"],
        "tgt_cols": ["title","manufacturer","price"],
    },
    "Walmart-Amazon":{
        "path": f"{BASE_DIR}/Walmart-Amazon",
        "src_cols": ["title","modelno","brand","category","price"],
        "tgt_cols": ["title","modelno","brand","category","price"],
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def clean(text):
    if pd.isna(text): return ""
    t = str(text).lower()
    t = re.sub(r'[^a-z0-9\s\.\-]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def load_records(cfg):
    src = pd.read_csv(os.path.join(cfg["path"],"source.csv")).fillna("")
    tgt = pd.read_csv(os.path.join(cfg["path"],"target.csv")).fillna("")
    s = src[[c for c in cfg["src_cols"] if c in src.columns]].apply(
            lambda r:" ".join(clean(v) for v in r), axis=1).tolist()
    t = tgt[[c for c in cfg["tgt_cols"] if c in tgt.columns]].apply(
            lambda r:" ".join(clean(v) for v in r), axis=1).tolist()
    return s, t

def load_pairs(cfg, n=N_PAIRS):
    path = cfg["path"]
    src  = pd.read_csv(os.path.join(path,"source.csv")).fillna("").set_index("id")
    tgt  = pd.read_csv(os.path.join(path,"target.csv")).fillna("").set_index("id")
    test = pd.read_csv(os.path.join(path,"test.csv"))
    pairs=[]
    for _,row in test.iterrows():
        l,r = row["ltable_id"], row["rtable_id"]
        if l not in src.index or r not in tgt.index: continue
        lt=" ".join(clean(src.loc[l,c]) for c in cfg["src_cols"] if c in src.columns)
        rt=" ".join(clean(tgt.loc[r,c]) for c in cfg["tgt_cols"] if c in tgt.columns)
        pairs.append((lt,rt))
        if len(pairs)>=n: break
    return pairs

# ──────────────────────────────────────────────────────────────────────────────
# Embedding 
# ──────────────────────────────────────────────────────────────────────────────
def measure_embedding(model_name, texts):
    print(f"     [Embedding] {model_name.split('/')[-1]}  ({len(texts)} records)")
    enc = SentenceTransformer(model_name, device="cpu")
    # warm-up
    enc.encode(texts[:8], batch_size=8, show_progress_bar=False,
               convert_to_numpy=True, device="cpu")
    times=[]
    emb=None
    for i in range(N_RUNS):
        t0=time.perf_counter()
        emb=enc.encode(texts, batch_size=64, show_progress_bar=False,
                        convert_to_numpy=True, device="cpu",
                        normalize_embeddings=False)
        t1=time.perf_counter()
        times.append(t1-t0)
        print(f"       run {i+1}/{N_RUNS}: {t1-t0:.2f}s")
    del enc; gc.collect()
    return float(np.mean(times)), np.ascontiguousarray(emb, dtype=np.float32)

# ──────────────────────────────────────────────────────────────────────────────
# Blocking 
# ──────────────────────────────────────────────────────────────────────────────
FAISS_WORKER = """
import sys, time, json, gc
import numpy as np
import faiss

src_path = sys.argv[1]
tgt_path = sys.argv[2]
out_path  = sys.argv[3]
n_runs    = int(sys.argv[4])

src = np.load(src_path)
tgt = np.load(tgt_path)

# C-contiguous float32
src = np.ascontiguousarray(src, dtype=np.float32)
tgt = np.ascontiguousarray(tgt, dtype=np.float32)
faiss.normalize_L2(src)
faiss.normalize_L2(tgt)
dim = src.shape[1]

times=[]
for _ in range(n_runs):
    t0=time.perf_counter()
    idx=faiss.IndexFlatIP(dim)
    idx.add(tgt)
    _D,_I=idx.search(src, 20)
    t1=time.perf_counter()
    del idx
    times.append(t1-t0)
    gc.collect()

result={"mean": float(np.mean(times)), "runs": times}
with open(out_path,"w") as f:
    json.dump(result, f)
print(f"DONE {result['mean']:.4f}s")
"""

def measure_blocking(src_emb, tgt_emb, ds_name):
    print(f"     [Blocking ] FAISS  ({len(src_emb)} src × {len(tgt_emb)} tgt)")
    times = []
    with tempfile.TemporaryDirectory() as tmp:
        src_path = os.path.join(tmp, "src.npy")
        tgt_path = os.path.join(tmp, "tgt.npy")
        out_path  = os.path.join(tmp, "result.json")
        worker_path = os.path.join(tmp, "faiss_worker.py")

        np.save(src_path, src_emb)
        np.save(tgt_path, tgt_emb)
        with open(worker_path, "w") as f:
            f.write(FAISS_WORKER)

        for i in range(N_RUNS):
            # Ξεκινάμε subprocess
            proc = subprocess.run(
                [sys.executable, worker_path,
                 src_path, tgt_path, out_path, "1"],
                capture_output=True, text=True, timeout=300
            )
            if proc.returncode != 0:
                print(f"       run {i+1} FAILED:\n{proc.stderr[-300:]}")
                return None
            with open(out_path) as f:
                res = json.load(f)
            t = res["runs"][0]
            times.append(t)
            print(f"       run {i+1}/{N_RUNS}: {t:.3f}s")

    mean = float(np.mean(times))
    print(f"     => Blocking mean: {mean:.3f}s")
    return mean

# ──────────────────────────────────────────────────────────────────────────────
# Matching
# ──────────────────────────────────────────────────────────────────────────────
def measure_matching(model_name, pairs):
    if not pairs: return None
    print(f"     [Matching ] {model_name.split('/')[-1]}  ({len(pairs)} pairs)")
    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, torch_dtype=torch.float32)
    model.eval()
    t1s=[p[0] for p in pairs]; t2s=[p[1] for p in pairs]
    enc=tok(t1s, t2s, truncation=True, max_length=128,
            padding=True, return_tensors="pt")
    # warm-up
    with torch.no_grad():
        b={k:v[:BATCH] for k,v in enc.items()}; model(**b)
    times=[]
    for i in range(N_RUNS):
        t0=time.perf_counter()
        with torch.no_grad():
            for s in range(0,len(pairs),BATCH):
                b={k:v[s:s+BATCH] for k,v in enc.items()}; model(**b)
        t1=time.perf_counter()
        times.append(t1-t0)
        print(f"       run {i+1}/{N_RUNS}: {t1-t0:.2f}s")
    del model; gc.collect()
    return float(np.mean(times))

# ──────────────────────────────────────────────────────────────────────────────
def main():
    results={}
    for model_name in MODELS:
        print(f"\n{'='*60}\nMODEL: {model_name}\n{'='*60}")
        results[model_name]={}
        for ds_name, cfg in DATASETS.items():
            print(f"\n  Dataset: {ds_name}")
            entry={}
            src_texts, tgt_texts = load_records(cfg)
            entry["n_source"]=len(src_texts)
            entry["n_target"]=len(tgt_texts)

            # Embedding
            try:
                emb_sec, embeddings = measure_embedding(
                    model_name, src_texts+tgt_texts)
                src_emb = embeddings[:len(src_texts)]
                tgt_emb = embeddings[len(src_texts):]
                entry["embedding_time_sec"]=round(emb_sec,3)
                print(f"     => Embedding: {emb_sec:.2f}s")
            except Exception as e:
                print(f"     => Embedding FAILED: {e}")
                entry["embedding_time_sec"]=None
                src_emb=tgt_emb=None

            # Blocking
            if src_emb is not None:
                try:
                    blk=measure_blocking(src_emb, tgt_emb, ds_name)
                    entry["blocking_time_sec"]=round(blk,4) if blk else None
                except Exception as e:
                    print(f"     => Blocking FAILED: {e}")
                    entry["blocking_time_sec"]=None
            else:
                entry["blocking_time_sec"]=None

            # Matching
            try:
                pairs=load_pairs(cfg)
                mch=measure_matching(model_name, pairs)
                entry["matching_time_sec"]=round(mch,3) if mch else None
                entry["n_matching_pairs"]=len(pairs)
                print(f"     => Matching: {mch:.2f}s")
            except Exception as e:
                print(f"     => Matching FAILED: {e}")
                entry["matching_time_sec"]=None

            results[model_name][ds_name]=entry
        gc.collect()

    out="timing_results.json"
    with open(out,"w") as f: json.dump(results,f,indent=2)
    print(f"\nΑποθηκεύτηκε: {out}")

    print(f"\n{'Model':<30} {'Dataset':<18} {'Emb':>8} {'Blk':>8} {'Mch':>8}")
    print("-"*72)
    for m,dd in results.items():
        short=m.split("/")[-1]
        for ds,v in dd.items():
            e=f"{v['embedding_time_sec']:.2f}s" if v.get('embedding_time_sec') else "ERR"
            b=f"{v['blocking_time_sec']:.3f}s"  if v.get('blocking_time_sec')  else "ERR"
            c=f"{v['matching_time_sec']:.2f}s"  if v.get('matching_time_sec')  else "ERR"
            print(f"{short:<30} {ds:<18} {e:>8} {b:>8} {c:>8}")

if __name__=="__main__":
    main()