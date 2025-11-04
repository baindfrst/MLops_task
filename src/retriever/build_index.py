import os
import json
import torch
import pickle
import logging
from time import perf_counter
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import normalize
from tqdm import tqdm

def setup_logging(output_dir: str = "artifacts"):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "build_index.log")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Log file is: {log_file}")
    return logger

def embed_batch(model, tok, texts, device, batch_size=64, logger=None):
    embs = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        t = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            emb = model(**t).last_hidden_state[:, 0, :]
            emb = normalize(emb, dim=-1)

        embs.append(emb.cpu())
        torch.cuda.empty_cache()

        if logger and (i // batch_size) % 10 == 0:
            logger.info(f"Обработано {i}/{total} текстов")

    return torch.vstack(embs)


def build_index(model_dir, passages_file, index_file, output_dir="artifacts"):
    logger = setup_logging(output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Model loaded from {model_dir}")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).to(device).eval()

    logger.info(f"Pass loaded from {passages_file}")
    passages = [json.loads(l) for l in open(passages_file, encoding="utf-8")]
    texts = [p["text"] for p in passages]
    pids = [p["pid"] for p in passages]
    logger.info(f"Docs: {len(passages)}")

    logger.info("Getting embd")
    embs = embed_batch(model, tok, texts, device, batch_size=32, logger=logger)

    os.makedirs(os.path.dirname(index_file), exist_ok=True)
    with open(index_file, "wb") as f:
        pickle.dump({"pids": pids, "embeddings": embs}, f)

    logger.info(f"Idx saved to: {index_file}")
    logger.info("idx build over")

if __name__ == "__main__":
    MODEL_DIR = "artifacts/retriever_ru"
    PASSAGES_FILE = "data/processed/passages.jsonl"
    INDEX_FILE = "artifacts/passages_index.pkl"

    build_index(MODEL_DIR, PASSAGES_FILE, INDEX_FILE)
