import os, json, argparse, yaml, numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import ndcg_score

import math

def compute_recall_at_k(scores: torch.Tensor, gold: list[int], k: int):
    """
    scores: [num_queries, num_docs] similarity matrix
    gold: index of true passage for each query
    """
    total = len(gold)
    hits = 0

    for i in range(total):
        topk = scores[i].topk(k).indices.tolist()
        if gold[i] in topk:
            hits += 1

    return hits / total

def compute_ndcg_at_k(scores: torch.Tensor, gold: list[int], k: int):
    """
    gold: index of relevant doc
    D = 1 for true, 0 for others
    """
    ndcgs = []
    for i in range(len(gold)):
        true = gold[i]
        ranking = scores[i].topk(k).indices.tolist()

        gains = [1 if doc == true else 0 for doc in ranking]
        discounts = [1/math.log2(idx+2) for idx in range(len(gains))]

        dcg = sum(g * d for g, d in zip(gains, discounts))
        idcg = 1

        ndcgs.append(dcg / idcg)

    return sum(ndcgs) / len(ndcgs)

def read_cfg(path: str): 
    with open(path, "r", encoding="utf-8") as f: 
        return yaml.safe_load(f)

def load_passages(path: str):
    pids, texts = [], []
    with open(path, "r", encoding="utf-8") as f:
        for l in f:
            j = json.loads(l); pids.append(j["pid"]); texts.append(j["text"])
    return pids, texts

def encode_texts(model, tok, texts, max_len=256, device="cpu", bs=64):
    embs = []
    for i in range(0, len(texts), bs):
        inp = tok(texts[i:i+bs], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with torch.no_grad():
            e = model(**inp).last_hidden_state[:,0,:]
        embs.append(e.cpu())
    return torch.cat(embs, dim=0).numpy()