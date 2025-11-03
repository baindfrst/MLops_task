from fastapi import FastAPI
from pydantic import BaseModel
import torch, pickle, json
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from torch.nn.functional import normalize

RETRIEVER_DIR = "artifacts/retriever_ru"
INDEX_FILE = "artifacts/passages_index.pkl"
PASSAGES_FILE = "data/processed/passages.jsonl"

LLM_NAME = "Qwen/Qwen2-1.5B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(RETRIEVER_DIR)
retriever = AutoModel.from_pretrained(RETRIEVER_DIR).to(device).eval()

with open(INDEX_FILE, "rb") as f:
    idx = pickle.load(f)
pids, emb_matrix = idx["pids"], idx["embeddings"]

pid2text = {json.loads(l)["pid"]: json.loads(l)["text"] for l in open(PASSAGES_FILE, encoding="utf-8")}

gen_tok = AutoTokenizer.from_pretrained(LLM_NAME)
gen_model = AutoModelForCausalLM.from_pretrained(
    LLM_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
gen_model.eval()

app = FastAPI()

class Query(BaseModel):
    text: str
    k: int = 3


def embed(text):
    t = tok([text], truncation=True, padding=True, return_tensors="pt", max_length=256).to(device)
    with torch.no_grad():
        emb = retriever(**t).last_hidden_state[:,0,:]
        emb = normalize(emb, dim=-1)
    return emb.cpu()

def retrieve(query, k=3):
    q_emb = embed(query)
    scores = (q_emb @ emb_matrix.T)[0]
    top = scores.topk(k)
    results = []
    for s, idx in zip(top.values, top.indices):
        pid = pids[idx]
        results.append({"text": pid2text[pid], "score": float(s)})
    return results

def llm_answer(question, context):
    prompt = f"""
Ты — полезный русскоязычный помощник. Используй контекст ниже.

Контекст:
{context}

Вопрос: {question}

Ответь понятно и по делу.
"""

    input_ids = gen_tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = gen_model.generate(
            **input_ids,
            max_new_tokens=256,
            temperature=0.2,
            top_p=0.9,
            do_sample=True
        )

    return gen_tok.decode(out[0], skip_special_tokens=True)


@app.post("/ask")
def ask(q: Query):
    docs = retrieve(q.text, q.k)
    context = "\n\n---\n\n".join(d["text"] for d in docs)
    ans = llm_answer(q.text, context)
    return {"answer": ans, "docs": docs}
