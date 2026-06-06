#!/usr/bin/env python3
"""
Compare: AraT5 abstractive ONLY vs Hybrid with Scaffold Fallback
Usage:
  With test data:  python compare.py --test_data data/splits/test.jsonl
  Demo mode:       python compare.py --demo
"""
import argparse
import json
import numpy as np
import torch
import re
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu
from bert_score import score as bert_score

from src.extractive import ExtractiveScaffoldBuilder
from src.abstractive import AbstractiveSummarizer
from src.utils import load_jsonl


# Demo data (built-in, no test.jsonl needed)
DEMO_ARTICLES = [
    {
        "article": "القاهرة - أعلن البنك المركزي المصري اليوم عن قراره بتثبيت أسعار الفائدة على الإيداع والإقراض لليلة واحدة عند 18.25% و19.25% على التوالي. وجاء القرار خلال اجتماع لجنة السياسة النقدية الذي عقد الخميس الماضي. وأوضح البنك في بيانه أن هذا القرار يأتي في إطار سعيه لتحقيق استقرار الأسعار والحد من التضخم الذي وصل إلى 33% في أبريل الماضي. وأضاف أن المؤشرات الاقتصادية تظهر بوادر تحسن في معدلات النمو.",
        "reference": "أعلن البنك المركزي المصري عن تثبيت أسعار الفائدة عند 18.25% و19.25% لمواجهة التضخم."
    },
]


def split_sentences(text):
    text = str(text).replace("\n", ". ")
    return [s.strip() for s in re.split(r"(?<=[.!?؟])\s+", text) if len(s.strip()) > 10]


def evaluate_metrics(refs, preds, device="cpu"):
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    r1 = [rouge.score(r, p)["rouge1"].fmeasure for r, p in zip(refs, preds)]
    r2 = [rouge.score(r, p)["rouge2"].fmeasure for r, p in zip(refs, preds)]
    rL = [rouge.score(r, p)["rougeL"].fmeasure for r, p in zip(refs, preds)]
    bleu = corpus_bleu(preds, [refs])
    _, _, F1 = bert_score(preds, refs, lang="ar", device=device, verbose=False)
    return {
        "rouge1": round(np.mean(r1), 4),
        "rouge2": round(np.mean(r2), 4),
        "rougeL": round(np.mean(rL), 4),
        "bleu": round(bleu.score, 4),
        "bertscore_f1": round(F1.mean().item(), 4),
    }


def run_demo(extractive_path, abstractive_path, device):
    """Run comparison on built-in demo article."""
    print("\n[DEMO MODE — Built-in article]")
    
    ext = ExtractiveScaffoldBuilder(extractive_path, device=device)
    abs_model = AbstractiveSummarizer(abstractive_path, device=device)
    
    item = DEMO_ARTICLES[0]
    article = item["article"]
    ref = item["reference"]
    
    sentences = split_sentences(article)
    scaffold, _ = ext.build_scaffold(sentences, top_k=3)
    
    stops = {"في","من","إلى","على","هذا","التي","و","أن","هو","هي","كان","تم","بعد","قبل","كل","بين","ما","لا","أو","لم","عن"}
    keywords = [w for w in article.split() if len(w) > 3 and w not in stops][:5]
    
    raw = abs_model.generate(article, scaffold, keywords, domain="news")
    
    # Quality check
    words = raw.split()
    is_quality = len(words) >= 4 and len(set(words))/len(words) >= 0.7 and len([c for c in raw if '\u0600' <= c <= '\u06FF']) >= 10
    hybrid = raw if is_quality else scaffold
    
    print(f"\nReference:     {ref}")
    print(f"Scaffold:      {scaffold}")
    print(f"AraT5 Raw:     {raw}")
    print(f"Hybrid Result: {hybrid}")
    print(f"\nQuality Gate:  {'PASS' if is_quality else 'FAIL -> fallback to scaffold'}")
    
    # Compute ROUGE
    metrics = evaluate_metrics([ref], [hybrid], device)
    print(f"\nROUGE-1: {metrics['rouge1']:.4f}")
    print(f"BERTScore F1: {metrics['bertscore_f1']:.4f}")


def run_full_eval(test_data, extractive_path, abstractive_path, num_samples, device):
    """Run comparison on test dataset."""
    print(f"Loading test data: {test_data}")
    records = load_jsonl(test_data)
    if num_samples:
        records = records[:num_samples]
    
    ext = ExtractiveScaffoldBuilder(extractive_path, device=device)
    abs_model = AbstractiveSummarizer(abstractive_path, device=device)
    
    refs, preds_abs, preds_hybrid, preds_scaffold = [], [], [], []
    quality_failures = 0
    
    for i, rec in enumerate(records):
        if i % 20 == 0:
            print(f"  {i}/{len(records)}...")
        
        sentences = rec.article_sentences if hasattr(rec, 'article_sentences') else split_sentences(rec.article)
        if len(sentences) < 3:
            continue
        
        scaffold, _ = ext.build_scaffold(sentences, top_k=3)
        
        stops = {"في","من","إلى","على","هذا","التي","و","أن","هو","هي","كان","تم","بعد","قبل","كل","بين","ما","لا","أو","لم","عن"}
        keywords = [w for w in (rec.article if hasattr(rec, 'article') else str(rec)) .split() if len(w) > 3 and w not in stops][:5]
        
        raw = abs_model.generate(rec.article if hasattr(rec, 'article') else str(rec), scaffold, keywords, domain=getattr(rec, 'domain', 'news'))
        
        words = raw.split()
        is_quality = len(words) >= 4 and len(set(words))/len(words) >= 0.7 and len([c for c in raw if '\u0600' <= c <= '\u06FF']) >= 10
        if not is_quality:
            quality_failures += 1
        
        refs.append(rec.summary if hasattr(rec, 'summary') else "")
        preds_abs.append(raw)
        preds_hybrid.append(raw if is_quality else scaffold)
        preds_scaffold.append(scaffold)
    
    # Evaluate
    m_abs = evaluate_metrics(refs, preds_abs, device)
    m_hyb = evaluate_metrics(refs, preds_hybrid, device)
    m_sca = evaluate_metrics(refs, preds_scaffold, device)
    
    print("\n" + "=" * 60)
    print("  RESULTS: Abstractive Reference Summaries")
    print("=" * 60)
    print(f"  {'Metric':<15} {'AraT5 Only':>12} {'Hybrid':>12} {'Scaffold':>12}")
    print("  " + "-" * 50)
    for metric in ["rouge1", "rouge2", "rougeL", "bleu", "bertscore_f1"]:
        print(f"  {metric:<15} {m_abs[metric]:>12.4f} {m_hyb[metric]:>12.4f} {m_sca[metric]:>12.4f}")
    
    print(f"\nQuality gate failures: {quality_failures}/{len(refs)} ({100*quality_failures/len(refs):.0f}%)")
    
    best = "ABSTRACTIVE" if m_abs["bertscore_f1"] > m_hyb["bertscore_f1"] else "HYBRID with fallback"
    print(f"\n>>> RECOMMENDATION: Use {best}")
    print(f"    Best BERTScore: {max(m_abs['bertscore_f1'], m_hyb['bertscore_f1']):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", help="Path to test.jsonl")
    parser.add_argument("--demo", action="store_true", help="Run demo mode (no test data needed)")
    parser.add_argument("--extractive_model", default="./model/extractive_easc")
    parser.add_argument("--abstractive_model", default="./model/abstractive/best")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    
    if not args.test_data and not args.demo:
        print("Usage:")
        print("  python compare.py --demo")
        print("  python compare.py --test_data data/splits/test.jsonl")
        return
    
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu"
    
    if args.demo:
        run_demo(args.extractive_model, args.abstractive_model, device)
    else:
        run_full_eval(args.test_data, args.extractive_model, args.abstractive_model, args.num_samples, device)


if __name__ == "__main__":
    main()