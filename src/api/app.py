from fastapi import FastAPI
from pydantic import BaseModel
import torch, pickle
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import normalize

MODEL_DIR = "artifacts/retriever_ru"
INDEX_FILE = "artifacts/passages_index.pkl"

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModel.from_pretrained(MODEL_DIR).to(device).eval()

with open(INDEX_FILE, "rb") as f:
    index = pickle.load(f)
pids, embeddings = index["pids"], index["embeddings"]

class Query(BaseModel):
    text: str
    k: int = 3

def embed(text):
    t = tok([text], padding=True, truncation=True, return_tensors="pt", max_length=256).to(device)
    with torch.no_grad():
        emb = model(**t).last_hidden_state[:,0,:]
        return normalize(emb, dim=-1).cpu()

@app.post("/search")
def search(q: Query):
    q_emb = embed(q.text)
    scores = (q_emb @ embeddings.T)[0]
    topk = scores.topk(q.k)
    result = [{"pid": pids[i], "score": float(topk.values[j])} for j, i in enumerate(topk.indices)]
    return result
