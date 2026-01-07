import json
import os
import pickle
import logging
from typing import Any, Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer
from torch.nn.functional import normalize

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class RetrieverHandler(BaseHandler):
    """
    TorchServe handler for a TorchScript retriever.
    Expects request JSON:
      {"query": "<text>", "k": 3}

    Loads tokenizer + passages_index.pkl + passages.jsonl from TorchServe model_dir
    (directory where MAR is extracted).
    """

    def __init__(self):
        super().__init__()
        self.device: Optional[torch.device] = None
        self.model = None  # torch.jit.ScriptModule
        self.tok = None    # HF tokenizer

        self.pids = None
        self.emb_matrix: Optional[torch.Tensor] = None
        self.pid2text: Dict[Any, str] = {}

        self.max_len = 256

        # Paths resolved at runtime from model_dir
        self.model_dir: Optional[str] = None
        self.tokenizer_dir: Optional[str] = None
        self.index_path: Optional[str] = None
        self.passages_path: Optional[str] = None

        self.initialized = False

    def initialize(self, context):
        self.manifest = context.manifest
        properties = context.system_properties

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_dir = properties.get("model_dir")
        if not self.model_dir:
            raise RuntimeError("TorchServe did not provide model_dir in system_properties")

        logger.warning(f"[HANDLER INIT] torch.cuda.is_available = {torch.cuda.is_available()}")
        logger.warning(f"[HANDLER INIT] device = {self.device}")
        logger.warning(f"[HANDLER INIT] model_dir = {self.model_dir}")

        if torch.cuda.is_available():
            try:
                logger.warning(f"[HANDLER INIT] cuda device name = {torch.cuda.get_device_name(0)}")
            except Exception:
                logger.warning("[HANDLER INIT] unable to read cuda device name")

        # --- Resolve bundled paths (relative to model_dir) ---
        self.index_path = os.path.join(self.model_dir, "passages_index.pkl")
        self.passages_path = os.path.join(self.model_dir, "passages.jsonl")

        # --- Load TorchScript model.pt from model_dir ---
        model_pt = os.path.join(self.model_dir, "model.pt")
        if not os.path.isfile(model_pt):
            raise FileNotFoundError(f"model.pt not found at: {model_pt}")

        self.model = torch.jit.load(model_pt, map_location=self.device).eval()

        # --- Load tokenizer ---
        # Preferred layout: <model_dir>/retriever_ru/
        # But your MAR currently contains tokenizer files directly in <model_dir>.
        preferred_tok_dir = os.path.join(self.model_dir, "retriever_ru")
        if os.path.isdir(preferred_tok_dir):
            self.tokenizer_dir = preferred_tok_dir
        else:
            # Fallback: tokenizer files are in the root of model_dir
            self.tokenizer_dir = self.model_dir

        logger.warning(f"[HANDLER INIT] tokenizer_dir = {self.tokenizer_dir}")

        # local_files_only=True prevents accidental HuggingFace Hub access
        try:
            self.tok = AutoTokenizer.from_pretrained(self.tokenizer_dir, local_files_only=True)
        except Exception as e:
            # Give a more helpful error if tokenizer files are missing
            raise RuntimeError(
                f"Failed to load tokenizer from: {self.tokenizer_dir}. "
                f"Expected files like tokenizer.json / tokenizer_config.json / vocab.txt to be present."
            ) from e

        # --- Load index ---
        if not os.path.isfile(self.index_path):
            raise FileNotFoundError(
                f"Index file not found at: {self.index_path}. "
                f"Make sure you passed it via --extra-files (passages_index.pkl)."
            )

        with open(self.index_path, "rb") as f:
            idx = pickle.load(f)

        self.pids = idx["pids"]
        emb = idx["embeddings"]

        if isinstance(emb, torch.Tensor):
            self.emb_matrix = emb.float()
        else:
            self.emb_matrix = torch.tensor(emb, dtype=torch.float32)

        # Keep embeddings on CPU (safe default)
        self.emb_matrix = self.emb_matrix.cpu()

        # --- Load passages ---
        if not os.path.isfile(self.passages_path):
            raise FileNotFoundError(
                f"Passages file not found at: {self.passages_path}. "
                f"Make sure you passed it via --extra-files (passages.jsonl)."
            )

        self.pid2text = {}
        with open(self.passages_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                j = json.loads(line)
                self.pid2text[j["pid"]] = j["text"]

        self.initialized = True
        logger.warning("[HANDLER INIT] done")

    def preprocess(self, data: List[Dict[str, Any]]) -> Tuple[str, int]:
        if not data or not isinstance(data, list):
            raise ValueError("Empty request payload")

        req = data[0]

        body = req.get("body", None)
        if body is None:
            body = req.get("data", None)

        if body is None:
            raise ValueError("Request must contain 'body' or 'data'")

        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8")

        if isinstance(body, str):
            body = body.strip()
            payload = json.loads(body) if body else {}
        elif isinstance(body, dict):
            payload = body
        else:
            raise ValueError(f"Unsupported request body type: {type(body)}")

        query = payload.get("query")
        if not query or not isinstance(query, str) or not query.strip():
            raise ValueError("Field 'query' is required and must be a non-empty string")

        k = payload.get("k", 3)
        try:
            k = int(k)
        except Exception:
            raise ValueError("Field 'k' must be an integer")

        if k <= 0:
            k = 1

        if self.pids is not None:
            k = min(k, len(self.pids))

        return query, k

    def inference(self, inputs: Tuple[str, int]) -> List[Dict[str, Any]]:
        if not self.initialized:
            raise RuntimeError("Handler is not initialized")

        query, k = inputs

        t = self.tok(
            [query],
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        input_ids = t["input_ids"].to(self.device)
        attention_mask = t["attention_mask"].to(self.device)

        with torch.no_grad():
            q_emb = self.model(input_ids, attention_mask)
            if isinstance(q_emb, (tuple, list)):
                q_emb = q_emb[0]
            q_emb = normalize(q_emb, dim=-1).cpu()

        scores = (q_emb @ self.emb_matrix.T)[0]
        top = torch.topk(scores, k=k)

        results: List[Dict[str, Any]] = []
        for s, i in zip(top.values, top.indices):
            i_int = int(i)
            pid = self.pids[i_int]
            results.append(
                {"pid": pid, "score": float(s), "text": self.pid2text.get(pid, "")}
            )

        return results

    def postprocess(self, inference_output: List[Dict[str, Any]]) -> List[Any]:
        return [inference_output]
