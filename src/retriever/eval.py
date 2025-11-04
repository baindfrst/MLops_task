import json
import torch
import argparse
import logging
import os
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import normalize
from src.eval.metrics import compute_recall_at_k, compute_ndcg_at_k

def setup_logging(output_dir: str = "artifacts"):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "eval.log")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Log started to: {log_file}")
    return logger


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        data = [json.loads(l) for l in f]
    return data


def embed_batch(model, tok, texts, device, batch_size=32, max_len=256, logger=None):
    embs = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        t = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            emb = model(**t).last_hidden_state[:, 0, :]
            emb = normalize(emb, dim=-1)

        embs.append(emb.cpu())
        torch.cuda.empty_cache()

        if logger and i % (batch_size * 10) == 0:
            logger.info(f"Proc {i}/{total} smpls")

    return torch.cat(embs, dim=0)


def evaluate(model_dir, valid_file, passages_file, k=5, output_dir="artifacts"):
    logger = setup_logging(output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loaded model from {model_dir}")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).to(device).eval()

    logger.info("Load data....")
    valid = load_jsonl(valid_file)
    passages = load_jsonl(passages_file)

    pid2text = {p["pid"]: p["text"] for p in passages}
    queries = [x["query"] for x in valid]
    gold_pids = [x["positive_pid"] for x in valid]
    docs = [pid2text.get(pid, "") for pid in gold_pids]

    logger.info(f"Val data: {len(valid)}")
    logger.info("Get embd")
    q_emb = embed_batch(model, tok, queries, device, batch_size=16, logger=logger)

    logger.info("documents embd")
    d_emb = embed_batch(model, tok, docs, device, batch_size=16, logger=logger)

    logger.info("Calc metrics")
    scores = q_emb @ d_emb.T
    recall = compute_recall_at_k(scores, list(range(len(gold_pids))), k)
    ndcg = compute_ndcg_at_k(scores, list(range(len(gold_pids))), k)

    logger.info(f"Results Recall@{k} = {recall:.4f}, nDCG@{k} = {ndcg:.4f}")
    logger.info("Eval over.")

    return {"recall": recall, "ndcg": ndcg}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/retriever_ru")
    ap.add_argument("--data", default="data/processed/retriever_valid.jsonl")
    ap.add_argument("--passages", default="data/processed/passages.jsonl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()

    evaluate(args.model, args.data, args.passages, args.k, args.out)
