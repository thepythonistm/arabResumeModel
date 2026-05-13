"""
Hybrid Pipeline: AraBERT Extractive → AraT5 Abstractive
With fallback to Enhanced Extractive when AraT5 fails.
"""

from typing import Literal

from .abstractive import AbstractiveSummarizer
from .extractive import ExtractiveSummarizer


class HybridArabicSummarizationPipeline:
    """
    Sequential pipeline:
    1. AraBERT extracts top-N key sentences
    2. AraT5 generates summary (with fallback to extractive)
    """

    def __init__(
        self,
        abstractive_model_path: str = "UBC-NLP/AraT5-base",
        extractive_model_name: str = "aubmindlab/bert-base-arabertv2"
    ):
        self.extractive = ExtractiveSummarizer(extractive_model_name)
        self.abstractive = AbstractiveSummarizer(abstractive_model_path)

    def _reorder_sentences(self, sentences: list, original_text: str) -> list:
        def get_position(sent):
            pos = original_text.find(sent)
            return pos if pos != -1 else float('inf')
        return sorted(sentences, key=get_position)

    def _join_sentences(self, sentences: list) -> str:
        cleaned = [s.strip().rstrip('.').strip() for s in sentences if s.strip()]
        return " . ".join(cleaned) + " ."

    def summarize(
        self,
        text: str,
        mode: Literal["extractive", "abstractive", "hybrid", "both"] = "hybrid",
        top_n: int = 3,
        max_length: int = 128,
        debug: bool = False
    ) -> dict:
        result = {}
        extractive_summary = ""  # ← INITIALIZE HERE to avoid the error

        # ========== EXTRACTIVE ==========
        if mode in ("extractive", "hybrid", "both"):
            try:
                raw_sentences = self.extractive.summarize(text, top_n=top_n)
                if isinstance(raw_sentences, str):
                    raw_sentences = [raw_sentences]
                
                ordered_sentences = self._reorder_sentences(raw_sentences, text)
                extractive_summary = self._join_sentences(ordered_sentences)
                result["extractive"] = extractive_summary
                
                if debug:
                    print(f"[DEBUG] Extractive: {extractive_summary[:100]}...")
            except Exception as e:
                print(f"❌ Extractive failed: {e}")
                extractive_summary = text[:200]  # Fallback to first 200 chars

        # ========== ABSTRACTIVE ==========
        if mode in ("abstractive", "both"):
            try:
                abstractive_summary = self.abstractive.summarize(text, max_length=max_length)
                # Validate output
                if len(abstractive_summary.strip()) < 10:
                    abstractive_summary = extractive_summary if extractive_summary else text[:200]
            except Exception:
                abstractive_summary = extractive_summary if extractive_summary else text[:200]
            
            result["abstractive"] = abstractive_summary

        # ========== HYBRID (with fallback) ==========
        if mode in ("hybrid", "both"):
            try:
                # Try AraT5 with guidance
                if extractive_summary:
                    hybrid_input = f"key points: {extractive_summary} article: {text}"
                else:
                    hybrid_input = text
                
                if debug:
                    print(f"[DEBUG] Hybrid input: {hybrid_input[:150]}...")
                
                hybrid_summary = self.abstractive.summarize(hybrid_input, max_length=max_length)
                
                # Validate: if garbage/empty, fallback to extractive
                if len(hybrid_summary.strip()) < 10 or any(bad in hybrid_summary for bad in ["🚫", "solar", "Malay"]):
                    raise ValueError("Invalid AraT5 output")
                
                result["hybrid"] = hybrid_summary
                
                if debug:
                    print(f"[DEBUG] Hybrid (AraT5): {hybrid_summary[:100]}...")
                    
            except Exception:
                # FALLBACK: Use extractive summary
                result["hybrid"] = extractive_summary if extractive_summary else text[:200]
                
                if debug:
                    print(f"[DEBUG] Hybrid (Fallback): {result['hybrid'][:100]}...")

        return result


def hybrid_summarize(
    text: str,
    model_path: str = "UBC-NLP/AraT5-base",
    mode: Literal["extractive", "abstractive", "hybrid", "both"] = "hybrid",
    **kwargs
) -> dict:
    """One-shot hybrid summarization."""
    pipeline = HybridArabicSummarizationPipeline(model_path)
    return pipeline.summarize(text, mode=mode, **kwargs)