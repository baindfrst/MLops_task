import os, json, argparse, random, yaml
from typing import List
import re

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^0-9a-zа-яё\- ]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def tokenize_words(text: str) -> List[str]:
    return text.split()

def chunk_text(text: str, chunk_tokens=256, overlap=64):
    text = clean_text(text)
    tokens = tokenize_words(text)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_tokens]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        i += max(1, chunk_tokens - overlap)
    return chunks

def main(cfg_path: str):
    cfg = read_yaml(cfg_path)
    random.seed(cfg["seed"])
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "sberquad_train.jsonl")
    dev_path = os.path.join(out_dir, "sberquad_dev.jsonl")
    assert os.path.exists(train_path), "start src.data.load_sberquad"
    rows = []
    for p in [train_path, dev_path]:
        with open(p, "r", encoding="utf-8") as f:
            rows += [json.loads(l) for l in f]
    passages = []
    for r in rows:
        for idx, ch in enumerate(chunk_text(r["context"], cfg["chunk_tokens"], cfg["chunk_overlap"])):
            passages.append({
                "pid": f'{r["id"]}_{idx}',
                "text": ch
            })
    with open(os.path.join(out_dir, "passages.jsonl"), "w", encoding="utf-8") as f:
        for psg in passages:
            f.write(json.dumps(psg, ensure_ascii=False) + "\n")
    pairs = [{"query": r["question"], "positive_pid": f'{r["id"]}_0'} for r in rows]
    random.shuffle(pairs)
    split = int(len(pairs) * cfg["train_ratio"])
    train, valid = pairs[:split], pairs[split:]
    with open(os.path.join(out_dir, "retriever_train.jsonl"), "w", encoding="utf-8") as f:
        for x in train: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "retriever_valid.jsonl"), "w", encoding="utf-8") as f:
        for x in valid: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"Saved passages={len(passages)}, train_pairs={len(train)}, valid_pairs={len(valid)} to {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
