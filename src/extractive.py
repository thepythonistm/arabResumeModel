"""Extractive sentence scoring using AraBERT.

Builds an extractive scaffold by classifying each sentence's
importance relative to the article summary.
"""

import warnings
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")


class ExtractiveScaffoldBuilder:
    """Extractive scaffold builder using fine-tuned AraBERT.

    Example:
        >>> builder = ExtractiveScaffoldBuilder("./model/extractive")
        >>> sentences = ["sentence 1.", "sentence 2.", "sentence 3."]
        >>> scaffold, scores = builder.build_scaffold(sentences, top_k=3)
        >>> print(scaffold)
    """

    def __init__(
        self,
        model_path: str = "aubmindlab/bert-base-arabertv02",
        device: str = "auto",
        max_length: int = 256,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_sentences(self, sentences: List[str]) -> List[float]:
        """Score each sentence for salience."""
        if not sentences:
            return []

        scores = []
        for sent in sentences:
            encoding = self.tokenizer(
                sent,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            outputs = self.model(**encoding)
            prob = torch.softmax(outputs.logits, dim=-1)[0][1].item()
            scores.append(prob)
        return scores

    def build_scaffold(
        self, sentences: List[str], top_k: int = 3
    ) -> Tuple[str, List[float]]:
        """Build extractive scaffold from top-K sentences."""
        scores = self.score_sentences(sentences)
        if not scores:
            return "", []

        top_indices = sorted(np.argsort(scores)[-top_k:][::-1])
        scaffold = " ".join([sentences[i] for i in top_indices])
        return scaffold, scores