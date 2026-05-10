"""
Abstractive Arabic Summarization using fine-tuned AraT5.
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class AbstractiveSummarizer:
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def summarize(
        self,
        text: str,
        max_length: int = 128,
        num_beams: int = 6,
        early_stopping: bool = True,
        no_repeat_ngram_size: int = 3,
        length_penalty: float = 1.2,
        min_length: int = 20,
    ) -> str:
        inputs = self.tokenizer(
            "summarize: " + text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=early_stopping,
                no_repeat_ngram_size=no_repeat_ngram_size,
                length_penalty=length_penalty,
                do_sample=False,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)