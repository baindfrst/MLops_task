import os
import argparse
import torch
from transformers import AutoModel, AutoTokenizer

class CLSWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]

def main(model_dir: str, out_path: str, max_len: int = 256):
    device = "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    base = AutoModel.from_pretrained(model_dir).to(device).eval()
    wrapper = CLSWrapper(base).to(device).eval()

    dummy = tok(["тест"], return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    input_ids = dummy["input_ids"].to(device)
    attention_mask = dummy["attention_mask"].to(device)

    traced = torch.jit.trace(wrapper, (input_ids, attention_mask))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    traced.save(out_path)
    print(f"Saved TorchScript model to: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="artifacts/retriever_ru")
    ap.add_argument("--out_path", default="model-store/model.pt")
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()
    main(args.model_dir, args.out_path, args.max_len)
