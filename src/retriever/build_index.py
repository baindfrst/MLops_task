import json, torch, pickle
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import normalize
from tqdm import tqdm

MODEL_DIR = "artifacts/retriever_ru"
PASSAGES_FILE = "data/processed/passages.jsonl"
INDEX_FILE = "artifacts/passages_index.pkl"

device = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModel.from_pretrained(MODEL_DIR).to(device).eval()

def embed_batch(texts, batch_size=64):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        t = tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model(**t).last_hidden_state[:,0,:]
            emb = normalize(emb, dim=-1)
        embs.append(emb.cpu())
        torch.cuda.empty_cache()
    return torch.vstack(embs)

passages = [json.loads(l) for l in open(PASSAGES_FILE, encoding="utf-8")]
texts = [p["text"] for p in passages]
pids = [p["pid"] for p in passages]

embs = embed_batch(texts, batch_size=32)

with open(INDEX_FILE, "wb") as f:
    pickle.dump({"pids": pids, "embeddings": embs}, f)

print(f"Saved index: {INDEX_FILE}")
