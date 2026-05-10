"""
Extractive Arabic Summarization using AraBERT embeddings.
"""

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer


class ExtractiveSummarizer:
    def __init__(self, model_name: str = "aubmindlab/bert-base-arabertv2", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _encode(self, texts: list) -> np.ndarray:
        embeddings = []
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            batch_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.extend(batch_emb)
        return np.array(embeddings)

    def summarize(self, text: str, top_n: int = 3) -> str:
        try:
            from nltk.tokenize import sent_tokenize
        except ImportError:
            raise ImportError("nltk is required. Run: pip install nltk")

        sentences = sent_tokenize(text)
        if len(sentences) <= top_n:
            return text

        embeddings = self._encode(sentences)
        sim_matrix = cosine_similarity(embeddings)
        scores = sim_matrix.sum(axis=1)
        ranked_indices = np.argsort(scores)[-top_n:]
        selected = sorted(ranked_indices)
        return " ".join([sentences[i] for i in selected])