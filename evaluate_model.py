"""
Evaluation Script for Arabic Summarization System
Tests Extractive (AraBERT), Abstractive (AraT5), and Hybrid (AraBERT→AraT5) modes
"""

import os
import sys
import random
import nltk

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("[Bootstrap] Downloading NLTK punkt_tab...")
    nltk.download('punkt_tab', quiet=True)

from src.pipeline import HybridArabicSummarizationPipeline
from src.evaluate import SummarizationEvaluator
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
import torch


def safe_evaluate(name, dataset, predict_fn, evaluator, num_samples=50, seed=42):
    random.seed(seed)
    
    if hasattr(dataset, "keys"):
        dataset = dataset["train"]
    
    total = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), total)
    
    samples = dataset.select(indices)
    
    def wrapped_predict(article):
        try:
            return predict_fn(article)
        except Exception as e:
            print(f"  ❌ Prediction failed: {e}")
            return ""  
    
    print(f"Evaluating {name} on {len(samples)} samples...")
    return evaluator.evaluate_dataset(samples, wrapped_predict, num_samples=len(samples))


def main():
    print("=" * 70)
    print("ARABIC SUMMARIZATION MODEL - EVALUATION")
    print("=" * 70)
    
    print("\n[1/4] Loading Hybrid Pipeline (AraBERT + AraT5)...")
    pipeline = HybridArabicSummarizationPipeline("./model")
    evaluator = SummarizationEvaluator()
    print("✅ Pipeline loaded")
    
    # ========== DEBUG: Verify model is actually fine-tuned ==========
    print("\n" + "=" * 70)
    print("[DEBUG] VERIFYING MODEL WEIGHTS")
    print("=" * 70)
    
    print(f"Model path: ./model")
    print(f"Model type: {type(pipeline.abstractive.model).__name__}")
    print(f"Model device: {pipeline.abstractive.model.device}")
    
    # Check if model weights changed from base AraT5
    # T5 architecture: encoder.block[0].layer[0].SelfAttention.q
    print("\nLoading base AraT5 for comparison...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained("UBC-NLP/AraT5-base")
    
    # T5 uses .encoder directly, not .model.encoder
    base_w = base_model.encoder.block[0].layer[0].SelfAttention.q.weight[0, :5]
    my_w = pipeline.abstractive.model.encoder.block[0].layer[0].SelfAttention.q.weight[0, :5]
    
    weights_match = torch.allclose(base_w, my_w)
    print(f"Weights match base model: {weights_match}")
    if weights_match:
        print("❌ WARNING: Your model is still the BASE AraT5, not fine-tuned!")
        print("   → Check if ./model contains the correct hybrid-trained weights")
    else:
        print("✅ Model weights are different from base (fine-tuned)")
    
    # ========== DEBUG: Quick generation test ==========
    print("\n" + "=" * 70)
    print("[DEBUG] QUICK GENERATION TEST")
    print("=" * 70)
    
    test_article = "القاهرة - أعلن البنك المركزي المصري اليوم عن قراره بتثبيت أسعار الفائدة على الإيداع والإقراض لليلة واحدة عند 18.25% و19.25% على التوالي."
    print(f"Input article: {test_article[:60]}...")
    
    try:
        result = pipeline.summarize(test_article, mode="hybrid", top_n=3, debug=True)
        print(f"\nExtractive output: {result['extractive'][:100]}...")
        print(f"Hybrid output: {result['hybrid'][:100]}...")
        
        if not result['hybrid'] or result['hybrid'].strip() == "":
            print("❌ Hybrid output is EMPTY!")
        elif len(result['hybrid']) < 10:
            print(f"❌ Hybrid output too short: '{result['hybrid']}'")
        else:
            print("✅ Hybrid generation works")
            
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== LOAD DATASETS ==========
    print("\n[Load] Fetching datasets...")
    try:
        easc = load_dataset("arbml/EASC", split="train")
        arasum = load_dataset("arbml/AraSum", split="train")
        print(f"   EASC: {len(easc)} samples | AraSum: {len(arasum)} samples")
    except Exception as e:
        print(f"❌ Failed to load datasets: {e}")
        sys.exit(1)
    
    # ========== EVALUATION ==========
    print("\n" + "=" * 70)
    print("[2/4] EXTRACTIVE MODE (AraBERT)")
    print("=" * 70)
    extractive_metrics = safe_evaluate(
        "Extractive", easc,
        lambda article: pipeline.summarize(article, mode="extractive")["extractive"],
        evaluator, num_samples=50
    )
    evaluator.print_report(extractive_metrics)
    
    print("\n" + "=" * 70)
    print("[3/4] HYBRID MODE (AraBERT → AraT5)")
    print("=" * 70)
    hybrid_metrics = safe_evaluate(
        "Hybrid", easc,
        lambda article: pipeline.summarize(article, mode="hybrid", top_n=3)["hybrid"],
        evaluator, num_samples=50
    )
    evaluator.print_report(hybrid_metrics)
    
    print("\n" + "=" * 70)
    print("[4/4] ABSTRACTIVE MODE (AraT5)")
    print("=" * 70)
    abstractive_metrics = safe_evaluate(
        "Abstractive", arasum,
        lambda article: pipeline.summarize(article, mode="abstractive")["abstractive"],
        evaluator, num_samples=50
    )
    evaluator.print_report(abstractive_metrics)
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Mode':<25} | {'ROUGE-1':>8} | {'ROUGE-2':>8} | {'ROUGE-L':>8} | {'BLEU':>8}")
    print("-" * 70)
    for label, metrics in [
        ("Extractive (AraBERT)", extractive_metrics),
        ("Hybrid (AraBERT→AraT5)", hybrid_metrics),
        ("Abstractive (AraT5)", abstractive_metrics),
    ]:
        print(f"{label:<25} | {metrics['rouge1']:>8.4f} | {metrics['rouge2']:>8.4f} | {metrics['rougeL']:>8.4f} | {metrics['bleu']:>8.4f}")
    
    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()