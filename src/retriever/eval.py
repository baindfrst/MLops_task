import json, torch, argparse
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import normalize
from src.eval.metrics import compute_recall_at_k, compute_ndcg_at_k


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def embed_batch(model, tok, texts, device, batch_size=32, max_len=256):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        t = tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)

        with torch.no_grad():
            emb = model(**t).last_hidden_state[:,0,:]
            emb = normalize(emb, dim=-1)

        embs.append(emb.cpu())
        torch.cuda.empty_cache()

    return torch.cat(embs, dim=0)


def evaluate(model_dir, valid_file, passages_file, k=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {model_dir} ...")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).to(device).eval()

    print("Loading data")
    valid = load_jsonl(valid_file)
    passages = load_jsonl(passages_file)

    pid2text = {p["pid"]: p["text"] for p in passages}
    queries = [x["query"] for x in valid]
    gold_pids = [x["positive_pid"] for x in valid]
    docs = [pid2text[pid] for pid in gold_pids]

    print("Embedding queries")
    q_emb = embed_batch(model, tok, queries, device, batch_size=16)

    print("Embedding docs")
    d_emb = embed_batch(model, tok, docs, device, batch_size=16)

    scores = q_emb @ d_emb.T

    recall = compute_recall_at_k(scores, list(range(len(gold_pids))), k)
    ndcg = compute_ndcg_at_k(scores, list(range(len(gold_pids))), k)

    print(f"Recall@{k}: {recall:.4f}")
    print(f"nDCG@{k}: {ndcg:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/retriever_ru")
    ap.add_argument("--data", default="data/processed/retriever_valid.jsonl")
    ap.add_argument("--passages", default="data/processed/passages.jsonl")
    args = ap.parse_args()

    evaluate(args.model, args.data, args.passages)
