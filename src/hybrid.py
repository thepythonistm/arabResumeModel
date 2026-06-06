"""End-to-end hybrid summarization pipeline.

Orchestrates extractive scaffolding and abstractive generation
for a complete summarization workflow.
"""

import warnings
from typing import Dict, List, Optional

import numpy as np
import torch

from .extractive import ExtractiveScaffoldBuilder
from .abstractive import AbstractiveSummarizer

warnings.filterwarnings("ignore")


class ArabicSummarizer:
    """End-to-end Arabic summarization pipeline.

    Combines extractive sentence scoring with abstractive generation
    for high-quality, grounded summaries.

    Example:
        >>> summarizer = ArabicSummarizer("./model/extractive", "./model/abstractive/best")
        >>> result = summarizer.summarize(article, domain="news", return_full=True)
        >>> print(result["summary"])
    """

    def __init__(
        self,
        extractive_model_path: str = "./model/extractive",
        abstractive_model_path: str = "./model/abstractive/best",
        device: str = "auto",
        top_k: int = 3,
        max_keywords: int = 5,
    ):
        self.top_k = top_k
        self.max_keywords = max_keywords

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[Pipeline] Initializing on {self.device}...")
        self.extractive = ExtractiveScaffoldBuilder(
            model_path=extractive_model_path, device=device
        )
        self.abstractive = AbstractiveSummarizer(
            model_path=abstractive_model_path, device=device
        )
        print("[Pipeline] Ready.")

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract content keywords from text."""
        stops = {
            "في", "من", "إلى", "على", "هذا", "التي", "و", "أن",
            "هو", "هي", "كان", "تم", "بعد", "قبل", "كل", "بين",
            "ما", "لا", "أو", "لم", "عن", "قد", "كلما",
        }
        words = text.split()
        keywords = [w for w in words if len(w) > 3 and w not in stops]
        from collections import Counter
        freq = Counter(keywords)
        return [w for w, _ in freq.most_common(self.max_keywords)]

    def _segment_sentences(self, text: str) -> List[str]:
        """Arabic sentence segmentation."""
        import re
        text = text.replace("\n", ". ")
        delimiters = re.compile(r"(?<=[.!?\u061F])\s+")
        return [s.strip() for s in delimiters.split(text) if len(s.strip()) > 10]

    def _is_quality_summary(self, summary: str) -> bool:
        """Validate abstractive output quality."""
        if not summary or len(summary.split()) < 4:
            return False
        words = summary.split()
        # Check for excessive repetition
        unique_ratio = len(set(words)) / len(words) if words else 0
        if unique_ratio < 0.7:
            return False
        # No word should appear more than twice
        from collections import Counter
        if any(c > 2 for c in Counter(words).values()):
            return False
        # Must contain Arabic characters
        arabic_chars = len([c for c in summary if '\u0600' <= c <= '\u06FF'])
        if arabic_chars < 10:
            return False
        return True

    def summarize(
        self,
        article: str,
        domain: str = "news",
        keywords: Optional[List[str]] = None,
        return_full: bool = False,
    ) -> Dict[str, str]:
        """Summarize an Arabic article.

        Args:
            article: Full article text in Arabic.
            domain: "news" or "procedural".
            keywords: Optional pre-defined keywords.
            return_full: If True, returns all intermediate outputs.

        Returns:
            Dict with "summary", and optionally "scaffold", "keywords",
            "guided_input", "sentence_scores".
        """
        sentences = self._segment_sentences(article)
        if not sentences:
            return {"summary": ""} if not return_full else {
                "summary": "", "scaffold": "", "keywords": []
            }

        scaffold, scores = self.extractive.build_scaffold(
            sentences, top_k=self.top_k
        )

        if keywords is None:
            keywords = self._extract_keywords(article)

        # Attempt abstractive generation
        summary = self.abstractive.generate(
            article=article,
            scaffold=scaffold,
            keywords=keywords,
            domain=domain,
        )

        # Quality gate: fall back to extractive scaffold if abstractive fails
        if not self._is_quality_summary(summary):
            print("[Pipeline] Abstractive quality check failed, using extractive scaffold.")
            summary = scaffold

        result = {"summary": summary}
        if return_full:
            guided = self.abstractive.build_guided_input(
                article, scaffold, keywords, domain
            )
            result.update({
                "scaffold": scaffold,
                "keywords": keywords,
                "guided_input": guided[:500] + "..." if len(guided) > 500 else guided,
                "sentence_scores": [round(s, 3) for s in scores[:len(sentences)]],
            })
        return result

    def summarize_batch(
        self,
        articles: List[str],
        domains: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """Summarize multiple articles."""
        if domains is None:
            domains = ["news"] * len(articles)
        results = []
        for article, domain in zip(articles, domains):
            try:
                results.append(self.summarize(article, domain=domain))
            except Exception as e:
                results.append({"summary": f"[ERROR: {str(e)[:50]}]"})
        return results

    def unload_models(self):
        """Free GPU memory."""
        import gc
        del self.extractive.model
        del self.abstractive.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Pipeline] Models unloaded.")