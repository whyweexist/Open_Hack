"""
Quantized LLM Engine — runs a small GGUF-quantized language model
(Qwen-2.5-0.5B-Instruct Q4_K_M) on CPU via llama-cpp-python.

The engine provides:
  • Raw text generation
  • Structured violation-classification prompt with JSON parsing
  • Robust fallback when the model is unavailable
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import (
    LLM_CONTEXT_LENGTH,
    LLM_MAX_TOKENS,
    LLM_N_THREADS,
    LLM_TEMPERATURE,
    MODELS_DIR,
    LLM_MODEL_FILE,
)

logger = logging.getLogger(__name__)

# ── Prompt templates (Qwen chat format) ──────────────────────────────
_SYSTEM_PROMPT = """\
You are a CCPA compliance analyst. Given CCPA statute sections and a
description of a business practice, determine whether the practice
violates any CCPA section.

RULES:
- Respond with ONLY a JSON object, no other text.
- Format: {"harmful": true/false, "articles": ["Section XXXX.XXX", ...]}
- If harmful is false, articles MUST be an empty list [].
- If harmful is true, articles MUST list every violated section.
- Consider the CCPA sections provided below as the authoritative source.
"""

_USER_TEMPLATE = """\
RELEVANT CCPA SECTIONS:
{context}

BUSINESS PRACTICE:
"{query}"

Step-by-step analysis:
1. Identify data practices described.
2. Match against the CCPA sections above.
3. Determine if any section is violated.
4. Return your answer as JSON ONLY.
"""


class QuantizedLLMEngine:
    """Thin wrapper for GGUF model inference via llama-cpp-python."""

    def __init__(self, model_dir: str | Path = MODELS_DIR):
        self.model_dir = Path(model_dir)
        self._llm = None
        self._available = False

    # ── Lazy loading ───────────────────────────────────────────────────
    def load(self) -> bool:
        """Load the GGUF model.  Returns True on success."""
        if self._llm is not None:
            return self._available

        model_path = self.model_dir / LLM_MODEL_FILE
        if not model_path.exists():
            # Try to find any .gguf file in the directory
            gguf_files = list(self.model_dir.glob("*.gguf"))
            if gguf_files:
                model_path = gguf_files[0]
            else:
                logger.warning("No GGUF model found in %s — LLM disabled", self.model_dir)
                self._available = False
                return False

        try:
            from llama_cpp import Llama

            logger.info("Loading GGUF model: %s", model_path.name)
            self._llm = Llama(
                model_path=str(model_path),
                n_ctx=LLM_CONTEXT_LENGTH,
                n_threads=LLM_N_THREADS,
                n_gpu_layers=0,          # CPU only
                verbose=False,
            )
            self._available = True
            logger.info("LLM loaded successfully")
        except Exception as exc:
            logger.error("Failed to load LLM: %s", exc)
            self._available = False

        return self._available

    @property
    def is_available(self) -> bool:
        return self._available

    # ── Raw generation ─────────────────────────────────────────────────
    def generate(
        self,
        prompt: str,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
    ) -> str:
        if not self._available or self._llm is None:
            return ""

        try:
            output = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["```", "\n\n\n"],
            )
            content = output["choices"][0]["message"]["content"]
            return content.strip()
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            return ""

    # ── Structured classification ──────────────────────────────────────
    def classify_violation(
        self, context: str, query: str
    ) -> Optional[Dict[str, Any]]:
        """
        Send context + query through the model and parse the JSON
        response.  Returns ``None`` if parsing fails.
        """
        prompt = _USER_TEMPLATE.format(context=context, query=query)
        raw = self.generate(prompt)

        if not raw:
            return None

        return self._parse_json(raw)

    # ── JSON extraction ────────────────────────────────────────────────
    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:
        """Best-effort JSON extraction from model output."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in the output
        match = re.search(r"\{[^}]+\}", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if "harmful" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass

        return None
