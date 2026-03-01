"""
Core CCPA Compliance Analyzer.

Loads the LLM, constructs prompts with RAG context, generates
predictions, and parses them into the required JSON format.
"""

import json
import re
import logging
import torch
from typing import Dict, List, Optional, Any

from app.config import (
    LLM_MODEL_NAME, HF_TOKEN,
    MAX_NEW_TOKENS, TEMPERATURE, TOP_P, REPETITION_PENALTY,
    VALID_SECTIONS,
)
from app.rag_engine import RAGEngine

logger = logging.getLogger(__name__)


# ── System prompt ────────────────────────────────────────────
SYSTEM_PROMPT = """You are a legal compliance expert specializing in the California Consumer Privacy Act (CCPA). Your task is to analyze a described business practice and determine if it violates any provision of the CCPA.

You MUST respond with ONLY a valid JSON object in this exact format:
{"harmful": true, "articles": ["Section 1798.xxx", ...]}
or
{"harmful": false, "articles": []}

Rules:
1. Set "harmful" to true ONLY if the described practice clearly violates a CCPA provision.
2. If harmful is true, list ALL violated CCPA sections in "articles". Use the format "Section 1798.xxx".
3. If harmful is false, "articles" MUST be an empty list [].
4. Only cite sections where there is a clear violation based on the statute text provided.
5. Do NOT include any explanatory text — output ONLY the JSON object.
6. Be thorough: a single practice may violate multiple sections.

Key CCPA sections and when they are violated:
- Section 1798.100: Violated when a business collects personal information without proper notice, uses it for undisclosed purposes, or collects more than necessary.
- Section 1798.105: Violated when a business refuses or fails to honor consumer deletion requests.
- Section 1798.106: Violated when a business refuses to correct inaccurate personal information.
- Section 1798.110: Violated when a business fails to disclose what personal information it collects when asked.
- Section 1798.115: Violated when a business fails to disclose what personal information is sold or shared when asked.
- Section 1798.120: Violated when a business sells or shares personal information without providing opt-out rights, or sells data of minors without proper consent.
- Section 1798.121: Violated when a business uses sensitive personal information beyond what is necessary without providing opt-out.
- Section 1798.125: Violated when a business discriminates against or retaliates against consumers who exercise their CCPA rights.
- Section 1798.130: Violated when a business fails to respond to consumer requests within 45 days or doesn't provide required methods for submitting requests.
- Section 1798.135: Violated when a business doesn't provide "Do Not Sell or Share" links or doesn't honor opt-out preference signals.
- Section 1798.150: Violated when a business fails to implement reasonable security measures, leading to unauthorized access or data breaches."""


def build_analysis_prompt(
    user_prompt: str,
    retrieved_sections: List[Dict],
) -> str:
    """
    Build the full prompt for the LLM, including retrieved CCPA context.
    """
    # Format retrieved sections
    context_parts = []
    for sec in retrieved_sections:
        page_info = f" (PDF page {sec['page']})" if sec.get("page") else ""
        context_parts.append(f"{sec['text']}{page_info}")

    context = "\n\n---\n\n".join(context_parts)

    user_message = (
        f"Relevant CCPA provisions:\n\n{context}\n\n"
        f"---\n\n"
        f"Analyze the following business practice for CCPA violations:\n"
        f"\"{user_prompt}\"\n\n"
        f"Respond with ONLY the JSON object:"
    )

    return user_message


class CCPAAnalyzer:
    """
    Main analyzer class. Manages the LLM and RAG engine,
    and provides the analyze() method for the API.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.rag_engine = RAGEngine()
        self._ready = False

    def initialize(self, rag_engine: Optional[RAGEngine] = None):
        """
        Load the LLM into GPU memory and prepare for inference.
        Called once at startup.
        """
        if rag_engine:
            self.rag_engine = rag_engine

        logger.info(f"Loading LLM: {LLM_MODEL_NAME}")

        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )

            # Configure 4-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                LLM_MODEL_NAME,
                trust_remote_code=True,
                token=HF_TOKEN,
            )

            # Load model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                token=HF_TOKEN,
                torch_dtype=torch.float16,
            )

            self.model.eval()
            self._ready = True
            logger.info("LLM loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            self._ready = False
            raise

    def analyze(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze a business practice description for CCPA violations.

        Args:
            prompt: Natural language description of a business practice

        Returns:
            Dict with 'harmful' (bool) and 'articles' (list of str)
        """
        if not self._ready:
            logger.error("Analyzer not ready")
            return {"harmful": False, "articles": []}

        try:
            # 1. Retrieve relevant CCPA sections
            retrieved = self.rag_engine.retrieve(prompt)

            # 2. Build the analysis prompt
            user_message = build_analysis_prompt(prompt, retrieved)

            # 3. Format for the model's chat template
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]

            # 4. Tokenize
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=6144,
            ).to(self.model.device)

            # 5. Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    repetition_penalty=REPETITION_PENALTY,
                    do_sample=True if TEMPERATURE > 0 else False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # 6. Decode only new tokens
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response_text = self.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip()

            logger.info(f"Raw LLM response: {response_text[:500]}")

            # 7. Parse and validate the response
            result = self._parse_response(response_text)

            logger.info(f"Analysis result: {result}")
            return result

        except Exception as e:
            logger.error(f"Analysis error: {e}", exc_info=True)
            return {"harmful": False, "articles": []}

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Robustly parse the LLM response into the required format.

        Handles:
          - Clean JSON
          - JSON embedded in markdown code blocks
          - JSON with extra text before/after
          - Malformed responses (fallback)
        """
        # Strategy 1: Direct JSON parse
        try:
            result = json.loads(response_text)
            return self._validate_result(result)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract JSON from code blocks
        code_block_match = re.search(
            r'```(?:json)?\s*(\{.*?\})\s*```',
            response_text,
            re.DOTALL,
        )
        if code_block_match:
            try:
                result = json.loads(code_block_match.group(1))
                return self._validate_result(result)
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find JSON object in text
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                return self._validate_result(result)
            except json.JSONDecodeError:
                pass

        # Strategy 4: Find JSON with nested arrays
        json_match = re.search(
            r'\{[^{}]*"harmful"[^{}]*"articles"\s*:\s*\[.*?\][^{}]*\}',
            response_text,
            re.DOTALL,
        )
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                return self._validate_result(result)
            except json.JSONDecodeError:
                pass

        # Strategy 5: Manual extraction
        harmful = None
        if re.search(r'"harmful"\s*:\s*true', response_text, re.IGNORECASE):
            harmful = True
        elif re.search(r'"harmful"\s*:\s*false', response_text, re.IGNORECASE):
            harmful = False

        articles = []
        article_matches = re.findall(
            r'Section\s+1798\.\d{3}(?:\.\d+)?',
            response_text,
        )
        if article_matches:
            articles = list(dict.fromkeys(article_matches))  # De-dup

        if harmful is not None:
            return self._validate_result({
                "harmful": harmful,
                "articles": articles,
            })

        # Final fallback: not harmful
        logger.warning(
            f"Could not parse LLM response, defaulting to not harmful: "
            f"{response_text[:200]}"
        )
        return {"harmful": False, "articles": []}

    def _validate_result(self, result: Dict) -> Dict[str, Any]:
        """
        Ensure the result conforms to the required format:
          - harmful is a boolean
          - articles is a list of valid section strings
          - if harmful=false, articles must be empty
          - if harmful=true, articles must not be empty
        """
        # Ensure harmful is a boolean
        harmful = result.get("harmful")
        if isinstance(harmful, str):
            harmful = harmful.lower() in ("true", "yes", "1")
        elif not isinstance(harmful, bool):
            harmful = bool(harmful)

        # Ensure articles is a list of valid strings
        articles = result.get("articles", [])
        if not isinstance(articles, list):
            articles = []

        # Validate and normalize section IDs
        valid_articles = []
        for article in articles:
            if not isinstance(article, str):
                continue
            # Normalize format
            article = article.strip()
            # Extract section number
            match = re.search(r'1798\.\d{3}(?:\.\d+)?', article)
            if match:
                normalized = f"Section {match.group(0)}"
                if normalized in VALID_SECTIONS:
                    valid_articles.append(normalized)

        # De-duplicate while preserving order
        seen = set()
        unique_articles = []
        for a in valid_articles:
            if a not in seen:
                seen.add(a)
                unique_articles.append(a)

        # Enforce consistency rules
        if harmful and not unique_articles:
            # LLM said harmful but didn't cite articles — try to infer
            # This shouldn't happen often with a good prompt
            logger.warning("harmful=true but no valid articles cited")
            harmful = False  # Safe fallback
            unique_articles = []
        elif not harmful:
            unique_articles = []

        return {
            "harmful": harmful,
            "articles": unique_articles,
        }

    @property
    def is_ready(self) -> bool:
        return self._ready