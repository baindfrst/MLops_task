import torch
import random
from torch.nn.functional import normalize
from transformers import AutoTokenizer, AutoModel

def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_tokenizer(model_path: str):
    return AutoTokenizer.from_pretrained(model_path)

def load_encoder(model_path: str, device: str):
    return AutoModel.from_pretrained(model_path).to(device)

def encode(tokenizer, texts, max_len, device):
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return {k: v.to(device) for k, v in tokens.items()}

def get_embeddings(model, tokens):
    with torch.no_grad():
        emb = model(**tokens).last_hidden_state[:, 0, :]
    return normalize(emb, dim=-1)

def info_nce_loss(q, p, n=None):
    import torch.nn.functional as F
    q = normalize(q, dim=-1)
    p = normalize(p, dim=-1)

    logits_pos = q @ p.T
    labels = torch.arange(q.size(0), device=q.device)

    if n is not None:
        n = normalize(n, dim=-1)
        logits_neg = q @ n.T
        logits = torch.cat([logits_pos, logits_neg], dim=1)
        return F.cross_entropy(logits, labels)

    return F.cross_entropy(logits_pos, labels)
