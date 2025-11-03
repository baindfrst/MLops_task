import json
from pathlib import Path
import pytest

@pytest.mark.parametrize("path", [
    "data/processed/passages.jsonl",
    "data/processed/retriever_train.jsonl",
    "data/processed/retriever_valid.jsonl",
])
def test_dataset_exists(path):
    assert Path(path).exists(), f"Missing file: {path}"

def test_passages_jsonl_format():
    path = Path("data/processed/passages.jsonl")
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            assert "pid" in obj, "PID missing"
            assert "text" in obj, "Text missing"
            assert isinstance(obj["pid"], str)
            assert isinstance(obj["text"], str)
            assert len(obj["text"]) > 0

def test_pairs_format():
    path = Path("data/processed/retriever_train.jsonl")
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            assert "query" in obj, "Query missing"
            assert "positive_pid" in obj, "Positive PID missing"
            assert isinstance(obj["query"], str)
            assert isinstance(obj["positive_pid"], str)
            assert len(obj["query"]) > 0
