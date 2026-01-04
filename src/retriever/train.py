import os, json, random, argparse, yaml
from typing import Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import logging

import mlflow
from pathlib import Path
import hashlib
import time


def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training.log")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Log init: {log_file}")
    return logger


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def flatten_cfg(cfg: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(flatten_cfg(v, key))
        else:
            out[key] = v
    return out


class PairDataset(Dataset):
    def __init__(self, train_file: str, passages_file: str, subset: int | None = None, negs: int = 4):
        self.rows = [json.loads(l) for l in open(train_file, encoding="utf-8")]
        if subset:
            self.rows = self.rows[:subset]

        self.pid2text = {}
        with open(passages_file, "r", encoding="utf-8") as f:
            for l in f:
                j = json.loads(l)
                self.pid2text[j["pid"]] = j["text"]

        self.pids = list(self.pid2text.keys())
        self.negs = negs

    def __len__(self):
        return len(self.rows)

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
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )


def info_nce(q_emb, p_emb, n_emb):
    import torch.nn.functional as F

    q = F.normalize(q_emb, dim=-1)
    p = F.normalize(p_emb, dim=-1)
    logits_pos = q @ p.T
    if n_emb is not None:
        n = F.normalize(n_emb, dim=-1)
        logits_neg = q @ n.T
        logits = torch.cat([logits_pos, logits_neg], dim=1)
        labels = torch.arange(q.size(0), device=q.device)
        return F.cross_entropy(logits, labels)
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits_pos, labels)


def train(cfg_path: str):
    cfg = read_cfg(cfg_path)
    logger = setup_logging(cfg.get("output_dir", "artifacts"))
    logger.info("Learn retr start")
    logger.info(f"Cfg: {cfg}")
    output_dir = cfg.get("output_dir", "artifacts")

    mlflow_cfg = cfg.get("mlflow", {}) if isinstance(cfg, dict) else {}
    tracking_uri = mlflow_cfg.get("tracking_uri", "")
    experiment_name = mlflow_cfg.get("experiment_name", "retriever_training")
    run_name = mlflow_cfg.get("run_name", f"retriever_{time.strftime('%Y%m%d_%H%M%S')}")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"mlflow tracking: {tracking_uri}")
    mlflow.set_experiment(experiment_name)
    mlflow.transformers.autolog()

    dvc_lock_hash = sha256_file("dvc.lock")

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    train_file = "data/processed/retriever_train.jsonl"
    passages_file = "data/processed/passages.jsonl"

    ds = PairDataset(
        train_file,
        passages_file,
        subset=cfg.get("train_subset"),
        negs=cfg["negatives_per_query"],
    )
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    logger.info(f"Ds len is: {len(ds)} smpls")

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    enc = AutoModel.from_pretrained(cfg["base_model"]).to(device)
    logger.info(f"Model {cfg['base_model']} loaded")

    opt = torch.optim.AdamW(enc.parameters(), lr=float(cfg["lr"]))
    total_steps = max(1, len(dl) * cfg["epochs"] // max(1, cfg.get("grad_accum", 1)))
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
    use_fp16 = (cfg.get("precision", "fp32") == "fp16")
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    enc.train()
    step = 0
    logger.info("Start learning now!!!")

    with mlflow.start_run(run_name=run_name):
        try:
            mlflow.log_params(flatten_cfg(cfg))
        except Exception as e:
            logger.warning(f"MLflow log_params failed (cfg too large or invalid types): {e}")
            core_params = {k: cfg.get(k) for k in [
                "seed", "base_model", "lr", "batch_size", "epochs",
                "negatives_per_query", "max_len_query", "max_len_passage",
                "precision", "grad_accum", "train_subset", "log_interval"
            ] if k in cfg}
            mlflow.log_params(core_params)

        mlflow.set_tag("device", device)
        mlflow.set_tag("project", "retriever")
        if dvc_lock_hash:
            mlflow.set_tag("dvc_lock_sha256", dvc_lock_hash)

        try:
            mlflow.log_artifact(cfg_path, artifact_path="configs")
        except Exception as e:
            logger.warning(f"MLflow log_artifact(cfg) failed: {e}")

        for epoch in range(cfg["epochs"]):
            logger.info(f"Epoch {epoch + 1}/{cfg['epochs']}")

            for batch in dl:
                q, pos, negs = batch
                q_in = encode(tok, list(q), cfg["max_len_query"]).to(device)
                p_in = encode(tok, list(pos), cfg["max_len_passage"]).to(device)
                flat_negs = [n for ns in negs for n in ns]
                n_in = encode(tok, flat_negs, cfg["max_len_passage"]).to(device)

                with torch.amp.autocast("cuda", enabled=use_fp16):
                    q_emb = enc(**q_in).last_hidden_state[:, 0, :]
                    p_emb = enc(**p_in).last_hidden_state[:, 0, :]
                    n_emb = enc(**n_in).last_hidden_state[:, 0, :]
                    loss = info_nce(q_emb, p_emb, n_emb) / cfg.get("grad_accum", 1)

                scaler.scale(loss).backward()

                if (step + 1) % cfg.get("grad_accum", 1) == 0:
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()
                    sch.step()

                if step % cfg["log_interval"] == 0:
                    loss_val = float(loss.item())
                    logger.info(f"Epoch {epoch} | Step {step} | Loss {loss_val:.4f}")
                    mlflow.log_metric("train_loss", loss_val, step=step)
                    mlflow.log_metric("epoch", float(epoch), step=step)
                    try:
                        mlflow.log_metric("lr", float(opt.param_groups[0]["lr"]), step=step)
                    except Exception:
                        pass

                step += 1

        os.makedirs(output_dir, exist_ok=True)
        enc.save_pretrained(output_dir)
        tok.save_pretrained(output_dir)
        logger.info(f"Model saved {output_dir}")
        logger.info("Train over")

        try:
            mlflow.log_artifacts(output_dir, artifact_path="model")
        except Exception as e:
            logger.warning(f"MLflow log_artifacts(model_dir) failed: {e}")

        training_log = os.path.join(output_dir, "training.log")
        if os.path.exists(training_log):
            try:
                mlflow.log_artifact(training_log, artifact_path="logs")
            except Exception as e:
                logger.warning(f"MLflow log_artifact(training.log) failed: {e}")

        if os.path.exists("dvc.lock"):
            try:
                mlflow.log_artifact("dvc.lock", artifact_path="dvc")
            except Exception as e:
                logger.warning(f"MLflow log_artifact(dvc.lock) failed: {e}")

        mlflow.log_metric("final_step", float(step))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    train(args.config)
