import os, json, argparse, random
from datasets import load_dataset
from huggingface_hub import login
login(os.getenv("HF_TOKEN"))
print(os.getenv("HF_TOKEN"))
def save_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main(output_dir="data/processed", val_ratio=0.1, seed=42):
    random.seed(seed)

    ds = load_dataset("kuznetsoffandrey/sberquad")

    data = ds["train"]

    def to_rows(split):
        rows=[]
        for i, ex in enumerate(split):
            if not ex.get("answers") or not ex["answers"].get("text"):
                continue
            
            rows.append({
                "id": ex.get("id", str(i)),
                "context": ex["context"],
                "question": ex["question"],
                "answer_text": ex["answers"]["text"][0],
                "answer_start": ex["answers"]["answer_start"][0],
            })
        return rows

    rows = to_rows(data)

    random.shuffle(rows)
    split = int(len(rows) * (1 - val_ratio))
    train_rows = rows[:split]
    dev_rows = rows[split:]

    save_jsonl(train_rows, os.path.join(output_dir, "sberquad_train.jsonl"))
    save_jsonl(dev_rows, os.path.join(output_dir, "sberquad_dev.jsonl"))

    print(f"  train: {len(train_rows)}")
    print(f"  dev:   {len(dev_rows)}")
    print(f"Saved : {output_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="data/processed")
    args = ap.parse_args()
    main(args.output_dir)