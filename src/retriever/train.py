import os, json, random, argparse, yaml, math
from typing import Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import logging
import os

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def read_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class PairDataset(Dataset):
    def __init__(self, train_file: str, passages_file: str, subset: int|None=None, negs:int=4):
        self.rows = [json.loads(l) for l in open(train_file, encoding="utf-8")]
        if subset: self.rows = self.rows[:subset]
        self.pid2text = {}
        with open(passages_file, "r", encoding="utf-8") as f:
            for l in f:
                j = json.loads(l); self.pid2text[j["pid"]] = j["text"]
        self.pids = list(self.pid2text.keys())
        self.negs = negs

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        q = r["query"]
        pos = self.pid2text.get(r["positive_pid"], "")
        negs = []
        for _ in range(self.negs):
            nid = random.choice(self.pids)
            while nid == r["positive_pid"]:
                nid = random.choice(self.pids)
            negs.append(self.pid2text[nid])
        return q, pos, negs

def encode(tokenizer, texts, max_len):
    return tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")

def info_nce(q_emb, p_emb, n_emb):
    import torch.nn.functional as F
    q = torch.nn.functional.normalize(q_emb, dim=-1)
    p = torch.nn.functional.normalize(p_emb, dim=-1)
    logits_pos = q @ p.T
    if n_emb is not None:
        n = torch.nn.functional.normalize(n_emb, dim=-1)
        logits_neg = q @ n.T
        logits = torch.cat([logits_pos, logits_neg], dim=1)
        labels = torch.arange(q.size(0), device=q.device)
        return F.cross_entropy(logits, labels)
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits_pos, labels)

def train(cfg_path: str):
    cfg = read_cfg(cfg_path)

    log_file = os.path.join(cfg.get("output_dir", "artifacts"), "training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="w", encoding="utf-8")
        ],
    )


    logger.info("Starting training")
    logger.info(f"Config: {cfg}")

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processed_dir = os.path.dirname("data/processed/retriever_train.jsonl")
    train_file = "data/processed/retriever_train.jsonl"
    passages_file = "data/processed/passages.jsonl"

    ds = PairDataset(train_file, passages_file, subset=cfg.get("train_subset"), negs=cfg["negatives_per_query"])
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    enc = AutoModel.from_pretrained(cfg["base_model"]).to(device)

    opt = torch.optim.AdamW(enc.parameters(), lr=float(cfg["lr"]))
    total_steps = max(1, len(dl) * cfg["epochs"] // max(1, cfg.get("grad_accum", 1)))
    sch = get_linear_schedule_with_warmup(opt, int(0.1*total_steps), total_steps)
    use_fp16 = (cfg.get("precision", "fp32") == "fp16")
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    enc.train()
    step = 0
    for epoch in range(cfg["epochs"]):
        for batch in dl:
            q, pos, negs = batch
            q_in = encode(tok, list(q), cfg["max_len_query"]).to(device)
            p_in = encode(tok, list(pos), cfg["max_len_passage"]).to(device)
            flat_negs = [n for ns in negs for n in ns]
            n_in = encode(tok, flat_negs, cfg["max_len_passage"]).to(device)

            with torch.amp.autocast("cuda", enabled=use_fp16):
                q_emb = enc(**q_in).last_hidden_state[:,0,:]
                p_emb = enc(**p_in).last_hidden_state[:,0,:]
                n_emb = enc(**n_in).last_hidden_state[:,0,:]
                loss = info_nce(q_emb, p_emb, n_emb) / cfg.get("grad_accum",1)

            scaler.scale(loss).backward()
            if (step+1) % cfg.get("grad_accum",1) == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()

            if step % cfg["log_interval"] == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
                logger.info(f"Epoch {epoch}, step {step}, loss = {loss.item():.4f}")
            step += 1

    os.makedirs(cfg["output_dir"], exist_ok=True)
    enc.save_pretrained(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])
    print(f"Saved retriever to {cfg['output_dir']}")
    logger.info(f"Model saved to {cfg['output_dir']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    train(args.config)
