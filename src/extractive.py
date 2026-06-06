"""Extractive sentence scoring using AraBERT.

Builds an extractive scaffold by classifying each sentence's
importance relative to the article summary.
"""

import os
import warnings
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")

# Base model for tokenizer and config (HuggingFace hub — has all files)
ARABERT_BASE = "aubmindlab/bert-base-arabertv02"


def _load_model_weights(model, model_path: str):
    """Load weights from local folder (supports .bin, .safetensors, or subfolder)."""
    # Check for PyTorch format
    bin_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        return

    # Check for Safetensors format
    st_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(st_path):
        from safetensors.torch import load_file
        state_dict = load_file(st_path)
        model.load_state_dict(state_dict)
        return

    # Check nested folder (Colab sometimes saves one level deeper)
    for sub in os.listdir(model_path):
        sub_path = os.path.join(model_path, sub)
        if os.path.isdir(sub_path):
            for fname in ["pytorch_model.bin", "model.safetensors"]:
                fp = os.path.join(sub_path, fname)
                if os.path.exists(fp):
                    if fname.endswith(".bin"):
                        state_dict = torch.load(fp, map_location="cpu", weights_only=True)
                    else:
                        from safetensors.torch import load_file
                        state_dict = load_file(fp)
                    model.load_state_dict(state_dict)
                    return

    raise FileNotFoundError(
        f"No model weights found in {model_path}. "
        "Expected pytorch_model.bin or model.safetensors"
    )


class ExtractiveScaffoldBuilder:
    """Extractive scaffold builder using fine-tuned AraBERT.

    Loads tokenizer from HuggingFace hub (has sentencepiece/spm files)
    and model weights from local fine-tuned folder.

    Handles models saved without config.json by loading config from HF hub
    and weights from local folder.

    Example:
        >>> builder = ExtractiveScaffoldBuilder("./model/extractive")
        >>> sentences = ["sentence 1.", "sentence 2.", "sentence 3."]
        >>> scaffold, scores = builder.build_scaffold(sentences, top_k=3)
        >>> print(scaffold)
    """

    def __init__(
        self,
        model_path: str = ARABERT_BASE,
        device: str = "auto",
        max_length: int = 256,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.max_length = max_length

        # Always load tokenizer from HF hub (has all required files)
        print(f"[Extractive] Loading tokenizer from: {ARABERT_BASE}")
        self.tokenizer = AutoTokenizer.from_pretrained(ARABERT_BASE)

        if model_path == ARABERT_BASE:
            # Using base pretrained model (no local fine-tuned weights)
            print(f"[Extractive] Loading base model from: {ARABERT_BASE}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                ARABERT_BASE
            ).to(self.device)
        else:
            # Using fine-tuned local weights
            # Auto-detect common subfolder patterns (HF Trainer saves to 'best/')
            actual_path = model_path
            config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(config_path):
                for sub in ["best", "final", "checkpoint-last"]:
                    candidate = os.path.join(model_path, sub)
                    if os.path.exists(os.path.join(candidate, "config.json")):
                        actual_path = candidate
                        break

            print(f"[Extractive] Loading fine-tuned weights from: {actual_path}")

            # Check if local folder has a valid config.json
            config_path = os.path.join(actual_path, "config.json")
            has_config = os.path.exists(config_path)

            if has_config:
                # Standard: config + weights both present
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    actual_path
                ).to(self.device)
            else:
                # Fallback: load config from HF hub, weights from local folder
                print(f"[Extractive] No config.json found — using base config from {ARABERT_BASE}")
                config = AutoConfig.from_pretrained(ARABERT_BASE)
                config.num_labels = 2  # binary classification
                self.model = AutoModelForSequenceClassification.from_config(config)
                _load_model_weights(self.model, actual_path)
                self.model = self.model.to(self.device)

        self.model.eval()

    @torch.no_grad()
    def score_sentences(self, sentences: List[str]) -> List[float]:
        """Score each sentence for salience (probability of being in summary)."""
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