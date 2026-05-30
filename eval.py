#!/usr/bin/env python3
"""Evaluate Arabic summarizer. Run: python eval.py --demo"""
import argparse
import json
from typing import List, Dict

from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu
from bert_score import score as bert_score
from src.hybrid import ArabicSummarizer

DEMO_ARTICLE = """القاهرة - أعلن البنك المركزي المصري اليوم عن قراره بتثبيت أسعار الفائدة على الإيداع والإقراض لليلة واحدة عند 18.25% و19.25% على التوالي. وجاء القرار خلال اجتماع لجنة السياسة النقدية الذي عقد الخميس الماضي. وأوضح البنك في بيانه أن هذا القرار يأتي في إطار سعيه لتحقيق استقرار الأسعار والحد من التضخم الذي وصل إلى 33% في أبريل الماضي."""
DEMO_REFERENCE = "أعلن البنك المركزي المصري عن تثبيت أسعار الفائدة عند 18.25% و19.25% لمواجهة التضخم."

def evaluate(refs: List[str], preds: List[str], device: str = "cpu") -> Dict:
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    r1, r2, rL = [], [], []
    for ref, pred in zip(refs, preds):
        s = rouge.score(ref, pred)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rL.append(s["rougeL"].fmeasure)
    bleu = corpus_bleu(preds, [refs])
    P, R, F1 = bert_score(preds, refs, lang="ar", device=device, verbose=False)
    return {
        "rouge1": round(sum(r1)/len(r1), 4),
        "rouge2": round(sum(r2)/len(r2), 4),
        "rougeL": round(sum(rL)/len(rL), 4),
        "bleu": round(bleu.score, 4),
        "bertscore_f1": round(F1.mean().item(), 4),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    parser.add_argument("--test_data", help="Path to test.jsonl")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--output", default="evaluation_report.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--extractive_model", default="./model/extractive")
    parser.add_argument("--abstractive_model", default="./model/abstractive/best")
    args = parser.parse_args()

    if not args.demo and not args.test_data:
        print("Usage:\n  python eval.py --demo\n  python eval.py --test_data data/test.jsonl")
        return

    print("Loading summarizer...")
    s = ArabicSummarizer(
        extractive_model_path=args.extractive_model,
        abstractive_model_path=args.abstractive_model,
        device=args.device,
    )

    if args.demo:
        print("\n[DEMO MODE]")
        print(f"Reference: {DEMO_REFERENCE}")
        pred = s.summarize(DEMO_ARTICLE, domain="news")["summary"]
        print(f"Predicted: {pred}")
        metrics = evaluate([DEMO_REFERENCE], [pred])
    else:
        from src.utils import load_jsonl
        records = load_jsonl(args.test_data)
        if args.num_samples:
            records = records[:args.num_samples]
        refs, preds = [], []
        for i, rec in enumerate(records):
            if i % 50 == 0:
                print(f"  {i}/{len(records)}...")
            preds.append(s.summarize(rec.article, domain=rec.domain)["summary"])
            refs.append(rec.summary)
        metrics = evaluate(refs, preds)

    print("\n" + "=" * 55)
    print("EVALUATION REPORT")
    print("=" * 55)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("=" * 55)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {args.output}")

if __name__ == "__main__":
    main()