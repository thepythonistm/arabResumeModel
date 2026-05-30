"""End-to-end hybrid summarization pipeline."""

import warnings
from typing import Dict, List, Optional

import numpy as np
import torch

from .extractive import ExtractiveScaffoldBuilder
from .abstractive import AbstractiveSummarizer

warnings.filterwarnings("ignore")


class ArabicSummarizer:

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

        print(f"Initializing pipeline on {self.device}...")
        self.extractive = ExtractiveScaffoldBuilder(
            model_path=extractive_model_path, device=device
        )
        # Abstractive model loaded but not used until retrained
        self.abstractive = AbstractiveSummarizer(
            model_path=abstractive_model_path, device=device
        )
        print("Pipeline ready.")

    def _extract_keywords(self, text: str) -> List[str]:
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
        import re
        text = text.replace("\n", ". ")
        delimiters = re.compile(r"(?<=[.!?\u061F])\s+")
        return [s.strip() for s in delimiters.split(text) if len(s.strip()) > 10]

    def summarize(
        self,
        article: str,
        domain: str = "news",
        keywords: Optional[List[str]] = None,
        return_full: bool = False,
    ) -> Dict[str, str]:
        sentences = self._segment_sentences(article)
        if not sentences:
            return {"summary": "", "scaffold": "", "keywords": []} if return_full else {"summary": ""}

        # Extractive scaffold (primary output — abstractive model needs retraining)
        scaffold, scores = self.extractive.build_scaffold(sentences, top_k=self.top_k)

        if keywords is None:
            keywords = self._extract_keywords(article)

        # Use extractive scaffold as summary
        summary = scaffold

        result = {"summary": summary}
        if return_full:
            result.update({
                "scaffold": scaffold,
                "keywords": keywords,
                "sentence_scores": [round(s, 3) for s in scores[:len(sentences)]],
            })
        return result

    def summarize_batch(self, articles: List[str], domains: Optional[List[str]] = None) -> List[Dict[str, str]]:
        if domains is None:
            domains = ["news"] * len(articles)
        return [self.summarize(a, domain=d) for a, d in zip(articles, domains)]