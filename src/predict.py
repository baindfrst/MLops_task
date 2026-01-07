import argparse
import csv
import json
import os
import pickle
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import normalize


DEFAULT_MODEL_DIR = os.getenv("RETRIEVER_DIR", "artifacts/retriever_ru")
DEFAULT_INDEX_FILE = os.getenv("INDEX_FILE", "artifacts/passages_index.pkl")
DEFAULT_PASSAGES_FILE = os.getenv("PASSAGES_FILE", "data/processed/passages.jsonl")


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def read_queries(input_path: str):
    if input_path.lower().endswith(".csv"):
        with open(input_path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            return [row["query"] for row in r if row.get("query")]
    elif input_path.lower().endswith(".jsonl"):
        rows = load_jsonl(input_path)
        qs = [r.get("query") for r in rows if r.get("query")]
        return qs


def load_pid2text(passages_file: str) -> Dict[str, str]:
    pid2text = {}
    with open(passages_file, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            pid2text[j["pid"]] = j["text"]
    return pid2text


def embed_texts(model, tok, texts: List[str], device: str, batch_size: int = 16, max_len: int = 256):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        t = tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with torch.no_grad():
            e = model(**t).last_hidden_state[:, 0, :]
            e = normalize(e, dim=-1)
        embs.append(e.cpu())
    return torch.cat(embs, dim=0)


def retrieve_topk(q_emb: torch.Tensor, emb_matrix: torch.Tensor, pids: List[str], pid2text: Dict[str, str], k: int):
    scores = q_emb @ emb_matrix.T
    top = torch.topk(scores, k=k, dim=1)
    results = []
    for i in range(q_emb.size(0)):
        row_pids = [pids[int(idx)] for idx in top.indices[i]]
        row_scores = [float(s) for s in top.values[i]]
        row_texts = [pid2text.get(pid, "") for pid in row_pids]
        results.append({"pids": row_pids, "scores": row_scores, "texts": row_texts})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True, help="path to input queries .csv | .jsonl format. Must contain \'query\'")
    ap.add_argument("--output_path", required=True, help="path to output predictions")
    ap.add_argument("--k", type=int, default=3, help="top-k passages to retriev")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()

    if not os.path.exists(DEFAULT_MODEL_DIR):
        raise FileNotFoundError(f"model not found: {DEFAULT_MODEL_DIR}")
    if not os.path.exists(DEFAULT_INDEX_FILE):
        raise FileNotFoundError(f"index not found: {DEFAULT_INDEX_FILE}")
    if not os.path.exists(DEFAULT_PASSAGES_FILE):
        raise FileNotFoundError(f"passages not found: {DEFAULT_PASSAGES_FILE}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    queries = read_queries(args.input_path)

    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL_DIR)
    model = AutoModel.from_pretrained(DEFAULT_MODEL_DIR).to(device).eval()

    with open(DEFAULT_INDEX_FILE, "rb") as f:
        idx = pickle.load(f)
    pids = idx["pids"]
    emb_matrix = idx["embeddings"]

    if isinstance(emb_matrix, torch.Tensor):
        emb_matrix = emb_matrix.float()
    else:
        emb_matrix = torch.tensor(emb_matrix, dtype=torch.float32)

    pid2text = load_pid2text(DEFAULT_PASSAGES_FILE)

    q_emb = embed_texts(model, tok, queries, device=device, batch_size=args.batch_size, max_len=args.max_len).float()
    results = retrieve_topk(q_emb, emb_matrix, pids, pid2text, k=args.k)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["query", "top_pids", "top_scores", "top_texts"])
        w.writeheader()
        for q, r in zip(queries, results):
            w.writerow({
                "query": q,
                "top_pids": json.dumps(r["pids"], ensure_ascii=False),
                "top_scores": json.dumps(r["scores"], ensure_ascii=False),
                "top_texts": json.dumps(r["texts"], ensure_ascii=False),
            })

    print(f"saved: {args.output_path}")


if __name__ == "__main__":
    main()
