import pytest
from pathlib import Path

from src.api.rag_app import retrieve, RETRIEVER_DIR

@pytest.mark.skipif(
    not Path(RETRIEVER_DIR).exists(),
    reason="retriever not found — skipping"
)
def test_retrieve_output_format():
    query = "что такое безумие"
    docs = retrieve(query, k=2)

    assert isinstance(docs, list)
    assert len(docs) == 2
    for d in docs:
        assert "text" in d
        assert "score" in d
        assert isinstance(d["text"], str)
        assert isinstance(d["score"], float)
