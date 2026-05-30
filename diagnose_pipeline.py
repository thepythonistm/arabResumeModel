#!/usr/bin/env python3
"""
Diagnostic script showing the 3 stages of the hybrid pipeline:
1. Extractive scaffold (AraBERT)
2. Raw abstractive output (AraT5) — before quality gate
3. Final output — after quality gate / fallback

Usage:
  Show all 3 stages:       python diagnose_pipeline.py
  Disable fallback:        python diagnose_pipeline.py --no-fallback
  Custom article:          python diagnose_pipeline.py --no-fallback --file article.txt
"""
import argparse
import sys

from src.extractive import ExtractiveScaffoldBuilder
from src.abstractive import AbstractiveSummarizer
from src.hybrid import ArabicSummarizer


TEST_ARTICLE = """القاهرة - أعلن البنك المركزي المصري اليوم عن قراره بتثبيت أسعار الفائدة على 
الإيداع والإقراض لليلة واحدة عند 18.25% و19.25% على التوالي. 
وجاء القرار خلال اجتماع لجنة السياسة النقدية الذي عقد الخميس الماضي. 
وأوضح البنك في بيانه أن هذا القرار يأتي في إطار سعيه لتحقيق استقرار الأسعار 
والحد من التضخم الذي وصل إلى 33% في أبريل الماضي. 
وأضاف أن المؤشرات الاقتصادية تظهر بوادر تحسن في معدلات النمو."""


def extract_keywords(text):
    """Simple keyword extraction (same logic as hybrid.py)."""
    stops = {
        "في", "من", "إلى", "على", "هذا", "التي", "و", "أن",
        "هو", "هي", "كان", "تم", "بعد", "قبل", "كل", "بين",
        "ما", "لا", "أو", "لم", "عن", "قد", "كلما",
    }
    words = text.split()
    keywords = [w for w in words if len(w) > 3 and w not in stops]
    from collections import Counter
    freq = Counter(keywords)
    return [w for w, _ in freq.most_common(5)]


def segment_sentences(text):
    """Sentence segmentation (same logic as hybrid.py)."""
    import re
    text = text.replace("\n", ". ")
    delimiters = re.compile(r"(?<=[.!?\u061F])\s+")
    return [s.strip() for s in delimiters.split(text) if len(s.strip()) > 10]


def check_quality(summary):
    """Same quality check as hybrid.py."""
    if not summary or len(summary.split()) < 4:
        return False, "Too short (< 4 words)"
    words = summary.split()
    unique_ratio = len(set(words)) / len(words) if words else 0
    if unique_ratio < 0.7:
        return False, f"Low unique word ratio ({unique_ratio:.2f})"
    from collections import Counter
    if any(c > 2 for c in Counter(words).values()):
        return False, "Word repeated > 2 times"
    arabic_chars = len([c for c in summary if '\u0600' <= c <= '\u06FF'])
    if arabic_chars < 10:
        return False, "Not enough Arabic characters"
    return True, "Passed"


def run_diagnostic(article, no_fallback=False, extractive_path="./model/extractive",
                   abstractive_path="./model/abstractive/best", device="auto"):

    print("=" * 70)
    print("HYBRID PIPELINE — DIAGNOSTIC REPORT")
    print("=" * 70)

    # ── Stage 0: Input ──────────────────────────────────────────────────
    print("\n📄 INPUT ARTICLE:")
    print(f"   {article[:120]}...")
    print(f"   Length: {len(article.split())} words")

    # ── Stage 1: Extractive (AraBERT) ──────────────────────────────────
    print("\n" + "─" * 70)
    print("🔍 STAGE 1: EXTRACTIVE SCAFFOLD (AraBERT)")
    print("─" * 70)

    builder = ExtractiveScaffoldBuilder(extractive_path, device=device)
    sentences = segment_sentences(article)
    scaffold, scores = builder.build_scaffold(sentences, top_k=3)
    keywords = extract_keywords(article)

    print(f"\n   Sentences found: {len(sentences)}")
    for i, (s, sc) in enumerate(zip(sentences, scores)):
        marker = " ✅" if s in scaffold else ""
        print(f"   [{i}] Score: {sc:.3f}{marker} | {s[:60]}...")

    print(f"\n   🔨 EXTRACTIVE SCAFFOLD:")
    print(f"   {scaffold}")

    # ── Stage 2: Abstractive (AraT5 + LoRA) ────────────────────────────
    print("\n" + "─" * 70)
    print("✍️  STAGE 2: RAW ABSTRACTIVE OUTPUT (AraT5 + LoRA)")
    print("   BEFORE quality gate / fallback")
    print("─" * 70)

    abs_model = AbstractiveSummarizer(abstractive_path, device=device)
    guided = abs_model.build_guided_input(article, scaffold, keywords, domain="news")
    raw_abstractive = abs_model.generate(article, scaffold, keywords, domain="news")

    print(f"\n   📝 GUIDED INPUT (what AraT5 sees):")
    print(f"   {guided[:200]}...")

    print(f"\n   ✍️  RAW ABSTRACTIVE OUTPUT:")
    print(f"   {raw_abstractive}")

    # ── Stage 3: Quality Gate ─────────────────────────────────────────
    print("\n" + "─" * 70)
    print("🛡️  STAGE 3: QUALITY GATE / FALLBACK DECISION")
    print("─" * 70)

    passed, reason = check_quality(raw_abstractive)
    print(f"\n   Quality check: {'✅ PASSED' if passed else '❌ FAILED'}")
    print(f"   Reason: {reason}")

    if passed:
        final = raw_abstractive
        used = "Abstractive (AraT5)"
    else:
        if no_fallback:
            final = raw_abstractive
            used = "Abstractive (FORCED — garbage kept)"
            print(f"\n   ⚠️  Fallback DISABLED (--no-fallback)")
        else:
            final = scaffold
            used = "Extractive Scaffold (AraBERT fallback)"
            print(f"\n   🔄 Fallback to extractive scaffold activated")

    print(f"\n   ✅ FINAL OUTPUT ({used}):")
    print(f"   {final}")

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 SUMMARY: Is the pipeline truly hybrid?")
    print("=" * 70)

    if passed and not no_fallback:
        status = "🟢 FULL HYBRID — AraT5 produced quality output, used directly"
    elif not passed and no_fallback:
        status = "🔴 ABSTRACTIVE ONLY (FORCED) — AraT5 output kept despite garbage"
    elif not passed and not no_fallback:
        status = "🟡 HYBRID WITH FALLBACK — AraT5 failed, AraBERT scaffold used"
    else:
        status = "?"

    print(f"\n   {status}")
    print(f"\n   Pipeline behavior:")
    print(f"     • Extractive model (AraBERT): ALWAYS runs — identifies key sentences")
    print(f"     • Abstractive model (AraT5):  ALWAYS runs — attempts rephrasing")
    print(f"     • Quality gate:                {'PASSED ✅' if passed else 'FAILED ❌ — fallback triggered'}")
    print(f"     • Final source:                {used}")
    print(f"\n   Note: A truly hybrid pipeline uses BOTH models every time.")
    print(f"   The extractive scaffold feeds into AraT5 as guidance signals.")
    print(f"   The fallback only changes which output is shown to the user.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-fallback", action="store_true",
                        help="Disable quality gate fallback (show raw AraT5 output)")
    parser.add_argument("--file", help="Path to text file containing article")
    parser.add_argument("--extractive_model", default="./model/extractive")
    parser.add_argument("--abstractive_model", default="./model/abstractive/best")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            article = f.read()
    else:
        article = TEST_ARTICLE

    run_diagnostic(
        article,
        no_fallback=args.no_fallback,
        extractive_path=args.extractive_model,
        abstractive_path=args.abstractive_model,
        device=args.device,
    )


if __name__ == "__main__":
    main()