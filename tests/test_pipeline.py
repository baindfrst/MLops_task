def test_repo_structure():
    import os
    expected = [
        "configs/data.yaml",
        "configs/train_retriever.yaml",
        "src/data/load_sberquad.py",
        "src/data/preprocess.py",
        "src/retriever/train.py",
        "src/eval/metrics.py",
        "requirements.txt",
        "README.md",
    ]
    for p in expected:
        assert os.path.exists(p), f"Missing: {p}"
