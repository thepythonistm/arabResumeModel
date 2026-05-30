"""Abstractive summarization using AraT5 with LoRA fine-tuning.

Generates summaries conditioned on extractive scaffolds,
keywords, and domain tags for grounded abstractive output.
"""

import warnings
from typing import Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

warnings.filterwarnings("ignore")


class AbstractiveSummarizer:
    """Abstractive summarizer using AraT5 + optional LoRA adapters.

    Example:
        >>> summarizer = AbstractiveSummarizer("./models/abstractive/best_merged")
        >>> summary = summarizer.generate(
        ...     article="full article text...",
        ...     scaffold="top sentences from extractive stage...",
        ...     keywords=["word1", "word2"],
        ...     domain="news"
        ... )
    """

    def __init__(
        self,
        model_path: str = "UBC-NLP/AraT5-base",
        device: str = "auto",
        max_input_length: int = 512,
        max_target_length: int = 128,
    ):
        """Initialize the abstractive summarizer.

        Args:
            model_path: Path to AraT5 model (merged or base HF ID).
            device: Device to run inference on ("cuda", "cpu", or "auto").
            max_input_length: Max tokens for encoder input.
            max_target_length: Max tokens for decoder output.
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load model — handles both merged and LoRA adapter checkpoints
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
        """Construct the guided input prompt for AraT5.

        Args:
            article: Full article text.
            scaffold: Extractive scaffold from AraBERT.
            keywords: List of keywords to inject (optional).
            domain: "news" or "procedural".
            max_keywords: Maximum number of keywords to include.

        Returns:
            Formatted prompt string for AraT5.
        """
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
        """Generate abstractive summary.

        Args:
            article: Full article text.
            scaffold: Extractive scaffold from AraBERT.
            keywords: Optional list of keywords for conditioning.
            domain: "news" or "procedural".
            num_beams: Number of beams for beam search.
            no_repeat_ngram_size: N-gram blocking to prevent repetition.
            length_penalty: Length penalty for beam search.
            early_stopping: Stop when all beams reach EOS.

        Returns:
            Generated summary string.
        """
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


def main():
    """CLI entry point for training the abstractive model."""
    import argparse
    import gc
    import math

    import numpy as np
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import Dataset
    from transformers import (
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    parser = argparse.ArgumentParser(description="Train abstractive AraT5 + LoRA")
    parser.add_argument("--train_data", required=True, help="Path to train.jsonl")
    parser.add_argument("--val_data", required=True, help="Path to val.jsonl")
    parser.add_argument("--extractive_model", required=True, help="AraBERT path")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.cuda.empty_cache()
    gc.collect()

    # Load data
    from utils import ArabicNormalizer, Record, SentenceProcessor, load_jsonl

    train_recs = load_jsonl(args.train_data)
    val_recs = load_jsonl(args.val_data)

    ABSTRACTIVE_MODEL = "UBC-NLP/AraT5-base"
    abs_tokenizer = AutoTokenizer.from_pretrained(ABSTRACTIVE_MODEL)

    # Build guided inputs (simplified — full implementation would use extractive model)
    def prepare_split(records, tokenizer, max_input=512, max_target=128):
        from tqdm import tqdm

        norm = ArabicNormalizer()
        sp = SentenceProcessor(norm)

        # Simple keyword extraction
        stops = {
            "في",
            "من",
            "إلى",
            "على",
            "هذا",
            "التي",
            "و",
            "أن",
            "هو",
            "هي",
        }

        processed = []
        for r in tqdm(records, desc="Preparing data"):
            # Build simple scaffold from extractive labels
            positive_idx = [i for i, l in enumerate(r.extractive_labels) if l == 1]
            if len(positive_idx) >= 3:
                selected = positive_idx[:3]
            else:
                selected = list(range(min(3, len(r.article_sentences))))
            scaffold = " ".join([r.article_sentences[i] for i in selected])

            # Keywords
            kw = [w for w in r.article.split() if len(w) > 3 and w not in stops][:5]
            kw_str = " ".join(kw)

            domain_tag = "<news>" if r.domain == "news" else "<procedural>"
            guided = f"summarize: {domain_tag} keywords: {kw_str} extract: {scaffold} article: {r.article}"

            model_inputs = tokenizer(
                guided, truncation=True, max_length=max_input, padding="max_length", return_tensors="pt"
            )
            labels_enc = tokenizer(
                r.summary, truncation=True, max_length=max_target, padding="max_length", return_tensors="pt"
            )

            labels = labels_enc["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100

            processed.append({
                "input_ids": model_inputs["input_ids"][0],
                "attention_mask": model_inputs["attention_mask"][0],
                "labels": labels[0],
            })
        return processed

    dataset_cache = f"{args.output_dir}_dataset.pt"
    if not __import__("os").path.exists(dataset_cache):
        train_data = prepare_split(train_recs, abs_tokenizer)
        val_data = prepare_split(val_recs, abs_tokenizer)
        torch.save({"train": train_data, "val": val_data}, dataset_cache)
    else:
        cache = torch.load(dataset_cache, map_location="cpu")
        train_data, val_data = cache["train"], cache["val"]

    class CachedDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    train_abs = CachedDataset(train_data)
    val_abs = CachedDataset(val_data)

    # Model + LoRA
    base_model = AutoModelForSeq2SeqLM.from_pretrained(ABSTRACTIVE_MODEL).to(device)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    abs_model = get_peft_model(base_model, lora_config)
    abs_model.gradient_checkpointing_enable()
    abs_model.config.use_cache = False

    trainable = sum(p.numel() for p in abs_model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable:,} params")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    common_args = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_steps=650,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=100,
        report_to="none",
        seed=42,
        fp16=not use_bf16,
        bf16=use_bf16,
        optim="adamw_torch",
        remove_unused_columns=False,
    )

    try:
        training_args = TrainingArguments(**common_args, eval_strategy="steps", eval_steps=1000)
    except TypeError:
        training_args = TrainingArguments(**common_args, evaluation_strategy="steps", eval_steps=1000)

    data_collator = DataCollatorForSeq2Seq(abs_tokenizer, model=abs_model, pad_to_multiple_of=8)

    trainer = Trainer(
        model=abs_model,
        args=training_args,
        train_dataset=train_abs,
        eval_dataset=val_abs,
        data_collator=data_collator,
    )

    steps_per_epoch = math.ceil(len(train_abs) / (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    print(f"Training {total_steps} steps...")

    trainer.train()

    # Save and merge
    trainer.save_model(f"{args.output_dir}/best")
    abs_tokenizer.save_pretrained(f"{args.output_dir}/best")

    print("Merging adapters...")
    base_clean = AutoModelForSeq2SeqLM.from_pretrained(ABSTRACTIVE_MODEL).to(device)
    merged = PeftModel.from_pretrained(base_clean, f"{args.output_dir}/best").merge_and_unload()
    merged.save_pretrained(f"{args.output_dir}/best_merged")
    abs_tokenizer.save_pretrained(f"{args.output_dir}/best_merged")
    print("Done!")


if __name__ == "__main__":
    main()