import os, json, argparse, random, yaml, re, logging
from pathlib import Path
from typing import List
from statistics import mean
from src.logging_setup import setup_logging


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
        chunk = tokens[i:i + chunk_tokens]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        i += max(1, chunk_tokens - overlap)
    return chunks


def validate_and_log_stats(input_path: str, logger: logging.Logger):
    logger.info(f"Checking: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f]

    n = len(lines)
    if n == 0:
        logger.warning("Empty file")
        return

    empty_q = sum(1 for x in lines if not x.get("question"))
    empty_c = sum(1 for x in lines if not x.get("context"))
    avg_q_len = mean(len(x.get("question", "")) for x in lines)
    avg_c_len = mean(len(x.get("context", "")) for x in lines)

    logger.info(f"total samples: {n}")
    logger.info(f"empty Q: {empty_q} ({empty_q/n:.2%})")
    logger.info(f"empty C: {empty_c} ({empty_c/n:.2%})")
    logger.info(f"mean len Q: {avg_q_len:.1f}")
    logger.info(f"mean len C: {avg_c_len:.1f}")

    required_keys = {"context", "question", "answers"}
    bad_rows = [i for i, r in enumerate(lines) if not required_keys.issubset(r)]
    if bad_rows:
        logger.warning(f"format errors: {bad_rows[:5]} (total {len(bad_rows)})")
    else:
        logger.info("data form correct")

    idx = [r.get("id") for r in lines if "id" in r]
    unique_idx = len(set(idx))
    logger.info(f"uniq idx: {unique_idx}/{len(idx)}")


def main(cfg_path: str):
    logger = setup_logging(level="INFO", logfile="artifacts/preprocess.log")
    cfg = read_yaml(cfg_path)
    random.seed(cfg["seed"])

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "sberquad_train.jsonl")
    dev_path = os.path.join(out_dir, "sberquad_dev.jsonl")

    assert os.path.exists(train_path), "run src.data.load_sberquad first"

    validate_and_log_stats(train_path, logger)
    validate_and_log_stats(dev_path, logger)

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
        for x in train:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    with open(os.path.join(out_dir, "retriever_valid.jsonl"), "w", encoding="utf-8") as f:
        for x in valid:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    logger.info(f"saved: passages {len(passages)}, train_pairs {len(train)}, valid_pairs {len(valid)} +++ {out_dir}")
    logger.info("preproc complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
