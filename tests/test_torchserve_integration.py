import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PT = PROJECT_ROOT / "model-store" / "model.pt"
HANDLER_PY = PROJECT_ROOT / "torchserve" / "handler.py"
TOKENIZER_DIR = PROJECT_ROOT / "artifacts" / "retriever_ru"
INDEX_PKL = PROJECT_ROOT / "artifacts" / "passages_index.pkl"
PASSAGES_JSONL = PROJECT_ROOT / "data" / "processed" / "passages.jsonl"


def _has_any_tokenizer_files(tokenizer_dir: Path):
    if not tokenizer_dir.is_dir():
        return False
    required = [
        "tokenizer.json",
        "tokenizer_config.json",
        "config.json",
    ]
    return all((tokenizer_dir / name).is_file() for name in required)


@pytest.mark.ci
def test_required_files_for_torchserve_bundle_exist():
    if not MODEL_PT.is_file():
        pytest.skip(f"model.pt is not available in this environment: {MODEL_PT}")

    assert HANDLER_PY.is_file(), f"Missing: {HANDLER_PY}"
    assert INDEX_PKL.is_file(), f"Missing: {INDEX_PKL}"
    assert PASSAGES_JSONL.is_file(), f"Missing: {PASSAGES_JSONL}"
    assert _has_any_tokenizer_files(TOKENIZER_DIR), f"Tokenizer dir missing or incomplete: {TOKENIZER_DIR}"


@pytest.mark.ci
def test_can_build_mar_if_archiver_available(tmp_path: Path):
    archiver = shutil.which("torch-model-archiver")
    if archiver is None:
        pytest.skip("torch-model-archiver not found in PATH")

    if not MODEL_PT.is_file():
        pytest.skip(f"model.pt is not available in this environment: {MODEL_PT}")

    out_dir = tmp_path / "model-store"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        archiver,
        "--model-name", "mymodel",
        "--version", "1.0",
        "--serialized-file", str(MODEL_PT),
        "--handler", str(HANDLER_PY),
        "--extra-files",
        ",".join([
            str(TOKENIZER_DIR),
            str(INDEX_PKL),
            str(PASSAGES_JSONL),
        ]),
        "--export-path", str(out_dir),
        "--force",
    ]

    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)

    if r.returncode != 0:
        raise AssertionError(
            "torch-model-archiver failed\n"
            f"cmd: {' '.join(cmd)}\n\nstdout:\n{r.stdout}\n\nstderr:\n{r.stderr}"
        )

    mar_path = out_dir / "mymodel.mar"
    assert mar_path.is_file(), f"mar was not created at: {mar_path}"

    with zipfile.ZipFile(mar_path, "r") as z:
        names = set(z.namelist())

    assert "model.pt" in names
    assert "handler.py" in names
    assert any(n.endswith("passages_index.pkl") for n in names)
    assert any(n.endswith("passages.jsonl") for n in names)
    assert (
        any(n.endswith("tokenizer.json") for n in names)
        and any(n.endswith("tokenizer_config.json") for n in names)
    ), f"Tokenizer files not found in mar: {sorted(list(names))[:50]}"
