"""
Evaluation utilities for Arabic summarization.
Computes ROUGE and BLEU scores.
"""

import numpy as np
from datasets import Dataset
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer


class SummarizationEvaluator:
    """
    Evaluates summarization quality using standard metrics.
    """

    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self.smooth = SmoothingFunction().method1  # ← For BLEU

    def evaluate_sample(self, reference: str, prediction: str) -> dict:
        """Score a single prediction."""
        rouge_scores = self.rouge.score(reference, prediction)
        
        # BLEU calculation
        ref_tokens = reference.split()
        pred_tokens = prediction.split()
        bleu = sentence_bleu(
            [ref_tokens], 
            pred_tokens, 
            smoothing_function=self.smooth
        )
        
        return {
            "rouge1": rouge_scores["rouge1"].fmeasure,
            "rouge2": rouge_scores["rouge2"].fmeasure,
            "rougeL": rouge_scores["rougeL"].fmeasure,
            "bleu": bleu,  # ← Added!
        }

    def evaluate_dataset(
        self, dataset: Dataset, predict_fn, num_samples: int = None
    ) -> dict:
        """
        Evaluate on a HuggingFace dataset.

        Args:
            dataset: Dataset with 'article' and 'summary' columns.
            predict_fn: Function that takes article text and returns summary.
            num_samples: Limit evaluation to N samples (None = all).

        Returns:
            Aggregated metrics dictionary.
        """
        samples = dataset.select(
            range(min(num_samples or len(dataset), len(dataset)))
        )

        rouge1, rouge2, rougeL, bleu = [], [], [], []

        for example in samples:
            prediction = predict_fn(example["article"])
            reference = example["summary"]
            scores = self.evaluate_sample(reference, prediction)

            rouge1.append(scores["rouge1"])
            rouge2.append(scores["rouge2"])
            rougeL.append(scores["rougeL"])
            bleu.append(scores["bleu"])

        return {
            "samples": len(rouge1),
            "rouge1": float(np.mean(rouge1)),
            "rouge2": float(np.mean(rouge2)),
            "rougeL": float(np.mean(rougeL)),
            "bleu": float(np.mean(bleu)),
        }

    @staticmethod
    def print_report(metrics: dict):
        """Pretty print evaluation results."""
        print("=" * 50)
        print("EVALUATION REPORT")
        print("=" * 50)
        print(f"Samples evaluated: {metrics['samples']}")
        print(f"ROUGE-1:  {metrics['rouge1']:.4f}")
        print(f"ROUGE-2:  {metrics['rouge2']:.4f}")
        print(f"ROUGE-L:  {metrics['rougeL']:.4f}")
        print(f"BLEU:     {metrics['bleu']:.4f}")  # ← Added!
        print("=" * 50)