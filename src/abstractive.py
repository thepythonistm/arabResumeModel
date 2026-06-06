"""Abstractive summarization using AraT5 with LoRA fine-tuning.

Generates summaries conditioned on extractive scaffolds,
keywords, and domain tags for grounded abstractive output.
"""

import os
import warnings
from typing import List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

warnings.filterwarnings("ignore")

# Base model for tokenizer (HuggingFace hub — has all tokenizer files)
ARAT5_BASE = "UBC-NLP/AraT5-base"


class AbstractiveSummarizer:
    """Abstractive summarizer using AraT5 + optional LoRA adapters.

    Auto-detects model format:
      - LoRA adapters (best/ + adapter_config.json): loads base + merges
      - Merged model (best_merged/): loads directly
    """

    def __init__(
        self,
        model_path: str = "UBC-NLP/AraT5-base",
        device: str = "auto",
        max_input_length: int = 512,
        max_target_length: int = 128,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        # CRITICAL FIX: Detect if local path has LoRA adapters or merged model
        adapter_config = os.path.join(model_path, "adapter_config.json")
        is_lora = os.path.exists(adapter_config)

        if is_lora:
            # LoRA adapters: load tokenizer from base HF model (has all files)
            print(f"[Abstractive] Loading tokenizer from: {ARAT5_BASE}")
            self.tokenizer = AutoTokenizer.from_pretrained(ARAT5_BASE)

            print(f"[Abstractive] Loading LoRA adapters from: {model_path}")
            model = AutoModelForSeq2SeqLM.from_pretrained(ARAT5_BASE).to(self.device)
            model = PeftModel.from_pretrained(model, model_path)
            print("[Abstractive] Merging adapters for faster inference...")
            self.model = model.merge_and_unload()
        else:
            # Merged model or base HF model: load directly
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)

        self.model.eval()

    def build_guided_input(
        self,
        article: str,
        scaffold: str,
        keywords: Optional[List[str]] = None,
        domain: str = "news",
        max_keywords: int = 5,
    ) -> str:
        """Construct the guided input prompt for AraT5."""
        kw = keywords or []
        kw_str = " ".join(kw[:max_keywords])
        domain_tag = "<news>" if domain == "news" else "<procedural>"
        extra = ""
        if domain == "procedural":
            extra = " steps:unknown"
        return (
            f"summarize: {domain_tag} "
            f"keywords: {kw_str}{extra} "
            f"extract: {scaffold} "
            f"article: {article}"
        )

    @torch.no_grad()
    def generate(
        self,
        article: str,
        scaffold: str,
        keywords: Optional[List[str]] = None,
        domain: str = "news",
        num_beams: int = 4,
        no_repeat_ngram_size: int = 3,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
    ) -> str:
        """Generate abstractive summary."""
        guided_input = self.build_guided_input(article, scaffold, keywords, domain)

        inputs = self.tokenizer(
            guided_input,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_length=self.max_target_length,
            num_beams=num_beams,
            early_stopping=early_stopping,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def merge_lora_weights(adapter_path: str, output_dir: str):
    """Merge LoRA adapters into base model for standalone inference.
    Run this ONCE in Colab after training to create best_merged/."""
    print(f"Merging LoRA weights from {adapter_path}...")
    base = AutoModelForSeq2SeqLM.from_pretrained(ARAT5_BASE)
    model = PeftModel.from_pretrained(base, adapter_path)
    merged = model.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)
    merged.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(ARAT5_BASE)
    tokenizer.save_pretrained(output_dir)
    print(f"Merged model saved to: {output_dir}")


# Backwards compatibility
load_abstractive = AbstractiveSummarizer
generate_summary = AbstractiveSummarizer.generate