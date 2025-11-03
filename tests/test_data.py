import json
from pathlib import Path

def test_passages_format():
    path = Path("data/processed/passages.jsonl")
    assert path.exists()

    with open(path, "r", encoding="utf-8") as f:
        line = json.loads(next(f))
        assert "pid" in line
        assert "text" in line
        assert isinstance(line["pid"], str)
        assert isinstance(line["text"], str)
        assert len(line["text"]) > 0
