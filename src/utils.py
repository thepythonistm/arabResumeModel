"""Utility classes and functions for data handling."""

import json
import re
import unicodedata
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass
class Record:
    """Single article-summary record with metadata."""
    id: str
    source: str
    domain: str
    article: str
    summary: str
    article_sentences: List[str]
    extractive_labels: List[int]
    guidance_signals: Dict
    article_length: int
    summary_length: int
    compression_ratio: float
    quality_score: float

    def to_dict(self) -> Dict:
        return asdict(self)


class ArabicNormalizer:
    """Normalizes Arabic text for consistent processing."""

    DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u0640]")
    KASHIDA = re.compile(r"[\u0640]")
    ZW = re.compile(r"[\u200B-\u200F\uFEFF\u2060-\u206F]")

    HAMZA = {
        "\u0623": "\u0627",
        "\u0625": "\u0627",
        "\u0622": "\u0627",
        "\u0624": "\u0648",
        "\u0626": "\u064A",
    }

    PUNCT = {
        "\u060C": ",",
        "\u061B": ";",
        "\u061F": "?",
        "\u00AB": '"',
        "\u00BB": '"',
        "\u201C": '"',
        "\u201D": '"',
        "\u2026": "...",
        "\u2013": "-",
        "\u2014": "-",
    }

    def normalize(self, text: str) -> str:
        """Apply full Arabic normalization pipeline."""
        if not text:
            return ""

        text = unicodedata.normalize("NFKC", text)
        text = self.ZW.sub("", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", " [URL] ", text)
        text = re.sub(r"\S+@\S+\.\S+", " [EMAIL] ", text)
        text = self.DIACRITICS.sub("", text)
        text = self.KASHIDA.sub("", text)

        for h, p in self.HAMZA.items():
            text = text.replace(h, p)
        for a, s in self.PUNCT.items():
            text = text.replace(a, s)

        return re.sub(r"\s+", " ", text).strip()

    def is_arabic(self, text: str, min_ratio: float = 0.5) -> bool:
        """Check if text contains sufficient Arabic characters."""
        if not text:
            return False
        ar = len(re.findall(r"[\u0600-\u06FF]", text))
        tot = len(text.replace(" ", ""))
        return (ar / tot) >= min_ratio if tot else False


class SentenceProcessor:
    """Splits text into sentences and generates extractive labels."""

    def __init__(self, normalizer: Optional[ArabicNormalizer] = None):
        self.norm = normalizer or ArabicNormalizer()

    def split(self, text: str) -> List[str]:
        """Split text into sentences using multiple delimiters."""
        text = self.norm.normalize(text).replace("\n", ". ")
        sents, cur = [], ""
        for ch in text:
            cur += ch
            if ch in ".!?\u061F\u3002\uff01":
                stripped = cur.strip()
                if len(stripped) > 10:
                    sents.append(stripped)
                cur = ""
        if cur.strip() and len(cur.strip()) > 10:
            sents.append(cur.strip())
        return sents

    def rouge_labels(self, sents: List[str], summary: str) -> List[int]:
        """Generate binary labels based on ROUGE-like word overlap."""
        sw = set(summary.split())
        scores = []
        for s in sents:
            w = set(s.split())
            scores.append(len(w & sw) / len(w) if w else 0)
        th = __import__("numpy").percentile(scores, 70) if scores else 0
        return [1 if sc >= th and sc > 0 else 0 for sc in scores]


def load_jsonl(path: str) -> List[Record]:
    """Load records from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(Record(**json.loads(line)))
    return records


def save_jsonl(records: List[Record], path: str) -> None:
    """Save records to a JSONL file."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")