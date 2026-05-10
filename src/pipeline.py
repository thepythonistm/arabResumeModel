"""
Hybrid Pipeline: AraBERT Extractive → AraT5 Abstractive
AraBERT selects key sentences, AraT5 generates summary from them.
"""

from typing import Literal

from .abstractive import AbstractiveSummarizer
from .extractive import ExtractiveSummarizer


class HybridArabicSummarizationPipeline:
    """
    Sequential pipeline:
    1. AraBERT extracts top-N key sentences
    2. AraT5 generates abstractive summary from extracted sentences only
    """
    
    def __init__(
        self,
        abstractive_model_path: str,
        extractive_model_name: str = "aubmindlab/bert-base-arabertv2"
    ):
        self.extractive = ExtractiveSummarizer(extractive_model_name)
        self.abstractive = AbstractiveSummarizer(abstractive_model_path)
    
    def _reorder_sentences(self, sentences: list, original_text: str) -> list:
        """Reorder sentences by their original position in text (not by importance)."""
        def get_position(sent):
            pos = original_text.find(sent)
            return pos if pos != -1 else float('inf')
        return sorted(sentences, key=get_position)
    
    def _join_sentences(self, sentences: list) -> str:
        """Join sentences with clear Arabic punctuation."""
        cleaned = [s.strip().rstrip('.').strip() for s in sentences]
        return " . ".join(cleaned) + " ."
    
    def summarize(
        self,
        text: str,
        mode: Literal["extractive", "abstractive", "hybrid", "both"] = "hybrid",
        top_n: int = 3,
        max_length: int = 128,
        debug: bool = False
    ) -> dict:
        """
        Modes:
        - 'extractive': AraBERT only
        - 'abstractive': AraT5 on full text
        - 'hybrid': AraBERT → AraT5 (AraT5 summarizes extracted sentences only)
        - 'both': Returns extractive + abstractive + hybrid
        """
        result = {}
        
        # ========== EXTRACTIVE ==========
        if mode in ("extractive", "hybrid", "both"):
            raw_sentences = self.extractive.summarize(text, top_n=top_n)
            
            # Ensure list format
            if isinstance(raw_sentences, str):
                raw_sentences = [raw_sentences]
            
            # Reorder + join
            ordered_sentences = self._reorder_sentences(raw_sentences, text)
            extractive_summary = self._join_sentences(ordered_sentences)
            
            result["extractive"] = extractive_summary
            
            if debug:
                print(f"[DEBUG] Extractive ({len(extractive_summary)} chars): {extractive_summary[:150]}...")
        
        # ========== ABSTRACTIVE (full text) ==========
        if mode in ("abstractive", "both"):
            result["abstractive"] = self.abstractive.summarize(
                text, max_length=max_length
            )
        
        # ========== HYBRID ==========
        if mode in ("hybrid", "both"):
            if debug:
                print(f"[DEBUG] Hybrid input ({len(extractive_summary)} chars): {extractive_summary[:150]}...")
            
            hybrid_summary = self.abstractive.summarize(
                extractive_summary,
                max_length=max_length
            )
            result["hybrid"] = hybrid_summary
            
            if debug:
                print(f"[DEBUG] Hybrid output ({len(hybrid_summary)} chars): {hybrid_summary[:150]}...")
        
        return result


def hybrid_summarize(
    text: str,
    model_path: str,
    mode: Literal["extractive", "abstractive", "hybrid", "both"] = "hybrid",
    **kwargs
) -> dict:
    """One-shot hybrid summarization."""
    pipeline = HybridArabicSummarizationPipeline(model_path)
    return pipeline.summarize(text, mode=mode, **kwargs)