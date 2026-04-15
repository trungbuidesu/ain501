"""MiniLM-L6 ONNX embedding via ONNX Runtime + HuggingFace tokenizer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.utils.artifact_paths import require_file
from src.utils.onnx_benchmark import create_session

DEFAULT_TOKENIZER = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MAX_LENGTH = 128


class MiniLMEmbedder:
    """Mean-pooled normalized embeddings matching `minilm_l6.py` ONNX export."""

    def __init__(
        self,
        model_path: Path | str,
        *,
        tokenizer_name: str = DEFAULT_TOKENIZER,
        max_length: int = DEFAULT_MAX_LENGTH,
        providers: list[str] | None = None,
        session: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.max_length = int(max_length)
        if session is not None:
            self._session = session
        else:
            require_file(
                self.model_path,
                model_id="minilm_l6",
                hint="Export or copy MiniLM-L6 ONNX; see configs/models/minilm_l6.yaml.",
            )
            self._session = create_session(
                self.model_path,
                providers=providers or ["CPUExecutionProvider"],
            )
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def encode(self, text: str) -> np.ndarray:
        """Return L2-normalized float32 vector (dim,) in the FAISS index space."""

        enc = self._tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )
        input_ids = enc["input_ids"].astype(np.int64)
        attention_mask = enc["attention_mask"].astype(np.int64)
        out = self._session.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )[0]
        vec = np.asarray(out, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(vec))
        if n > 1e-12:
            vec = vec / n
        return vec.astype(np.float32, copy=False)
