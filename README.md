# Arabic Text Summarization System

Fine-tuned AraT5 model for abstractive Arabic summarization, with AraBERT-based extractive fallback. Trained on the AraSum dataset (153K Arabic articles).

---

## Model Performance

| Metric | Value |
|--------|-------|
| Training Steps | 4,500 |
| Final Training Loss | 22.73 |
| Final Validation Loss | 4.93 |
| Architecture | AraT5-base (Seq2Seq) |
| Dataset | AraSum |

---

## Installation

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"

Python: 3.8+
GPU: Optional (CUDA recommended for faster inference)

Quick Start
1. Download the Model
Download the fine-tuned model from the link below and extract it to your project folder:
plain
Copy
your_project/
├── model/          ← Put the downloaded model here
└── src/
2. Run Summarization
Python
Copy
from src.pipeline import ArabicSummarizationPipeline

# Load pipeline
pipeline = ArabicSummarizationPipeline("./model")

# Summarize text
article = "نص عربي طويل هنا..."
result = pipeline.summarize(article, mode="both")

print("Extractive:", result["extractive"])
print("Abstractive:", result["abstractive"])
3. Modes
Table
Mode	Description	Use Case
extractive	Picks key sentences from original text	Fast, factually grounded
abstractive	Generates new summary sentences	Fluent, human-like
both	Returns both summaries	Comparison / fallback
#Model Download
The fine-tuned model weights (~2.5 GB) are available here:

#Project Structure
├── src/
│   ├── extractive.py      # AraBERT extractive summarizer
│   ├── abstractive.py     # AraT5 abstractive summarizer
│   ├── pipeline.py        # Combined pipeline
│   └── evaluate.py        # ROUGE & BLEU evaluation
├── notebooks/
│   └── training.ipynb     # Training notebook with logs
├── samples/
│   └── sample_output.json # Example input/output
├── requirements.txt
└── README.md
#Evaluation:
To evaluate the model on sample articles:
from src.pipeline import ArabicSummarizationPipeline
from src.evaluate import SummarizationEvaluator
from datasets import load_dataset

pipeline = ArabicSummarizationPipeline("./model")
evaluator = SummarizationEvaluator()
ds = load_dataset("arbml/AraSum")

metrics = evaluator.evaluate_dataset(
    ds["train"],
    lambda x: pipeline.summarize(x, mode="abstractive")["abstractive"],
    num_samples=50
)

evaluator.print_report(metrics)

#Sample Output:

  "article": "بدأت اليوم الجمعة( 23 أيلول/ سبتمبر 2016 ) في ميونيخ محاكمة رجل من جمهورية الجبل الأسود...",
  "reference_summary": "بدأت محكمة ميونخ النظر في اتهامات متعلقة برجل من جمهورية الجبل الأسود حاول نقل أسلحة إلى فرنسا...",
  "generated_summary": "كشفت الشرطة الألمانية أن رجل من دول القرن الأسود احتجزتها الشرطة في ألمانيا بتهمة \"داعش\"..."
 #sample 2:

ORIGINAL ARTICLE:

"بدأت اليوم الجمعة( 23 أيلول/ سبتمبر 2016 ) في ميونيخ محاكمة رجل من جمهورية الجبل الأسود ( مونتينيغرو)، اعتقل في ألمانيا للاشتباه في انه كان ينقل أسلحة قبل أيام من اعتداءات باريس التي حصلت في تشرين الثاني/نوفمبر العام الماضي. وتريد المحكمة معرفة عما إذا كانت هذه الأسلحة أعدت للاستخدام في هجمات فرنسا. وقالت المتحدثة إن المتهم ""اعترف أنه كان على علم بوجود أسلحة في سيارته، لكنه لا يعرف ما إذا كانت ستستخدم لتنفيذ اعتداء"". وأضافت المتحدثة أن الرجل الذي يبلغ الحادية والخمسين ويدعى فوسيليتش، ملاحق ""...


REFERENCE SUMMARY (Human-written):

بدأت محكمة ميونخ النظر في اتهامات متعلقة برجل من جمهورية الجبل الأسود حاول نقل أسلحة إلى فرنسا، وقبضت الشرطة الألمانية على الرجل بالقرب من الحدود النمساوية، حيث أعترف بعلمه بنقل الأسلحة إلا أنه لم يكن يعرف السبب.


GENERATED SUMMARY (Your Model):

"أعلنت النيابة العامة في ميونيخ أن رجل من جمهورية الجبل الأبيض ""مونتنيغرو"" الألماني احتجزته السلطات الألمانية بتهمة التخابر مع تنظيم ""داعش"" في هجوم باريس الذي وقع في نوفمبر/نوفمبر الماضي."

### 1. Download the Model
Download the fine-tuned model from the link below and extract it to your project folder:
The fine-tuned model weights (~2.5 GB) are available here:
https://drive.google.com/drive/folders/151tVfp8ecZUwYXQBNFPl3RqzPd7DFekk?usp=sharing

├── src/
│   ├── __init__.py          # Package exports
│   ├── extractive.py        # AraBERT extractive summarizer
│   ├── abstractive.py       # AraT5 abstractive summarizer
│   ├── pipeline.py          # Hybrid pipeline (AraBERT → AraT5)
│   └── evaluate.py          # ROUGE & BLEU evaluation
├── notebooks/
│   └── training.ipynb       # Training notebook with logs
├── samples/
│   └── sample_output.json   # Example input/output
├── test.py                  # Quick test script
├── evaluate_model.py        # Full evaluation script
├── requirements.txt
└── README.md
PS C:\Users\hp\arabic_summarization_deliverable> python evaluate_model.py

ARABIC SUMMARIZATION MODEL - EVALUATION


[1/4] Loading Hybrid Pipeline (AraBERT + AraT5)...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|████████████████████████████████████████| 199/199 [00:00<00:00, 2810.90it/s]
[transformers] BertModel LOAD REPORT from: aubmindlab/bert-base-arabertv2
Key                                        | Status     |  | 
-------------------------------------------+------------+--+-
cls.predictions.transform.dense.bias       | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 
cls.seq_relationship.weight                | UNEXPECTED |  | 
cls.predictions.transform.dense.weight     | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 
cls.seq_relationship.bias                  | UNEXPECTED |  | 
cls.predictions.bias                       | UNEXPECTED |  | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Loading weights: 100%|████████████████████████████████████████| 284/284 [00:00<00:00, 2613.06it/s]
[transformers] The tied weights mapping and config for this model specifies to tie shared.weight to lm_head.weight, but both are present in the checkpoints with different values, so we will NOT tie them. You should update the config with `tie_word_embeddings=False` to silence this warning.
[transformers] The tied weights mapping and config for this model specifies to tie shared.weight to encoder.embed_tokens.weight, but both are present in the checkpoints with different values, so we will NOT tie them. You should update the config with `tie_word_embeddings=False` to silence this warning.
[transformers] The tied weights mapping and config for this model specifies to tie shared.weight to decoder.embed_tokens.weight, but both are present in the checkpoints with different values, so we will NOT tie them. You should update the config with `tie_word_embeddings=False` to silence this warning.
✅ Pipeline loaded

[Load] Fetching datasets...
   EASC: 153 samples | AraSum: 49603 samples
   EASC columns: ['article', 'summary']
   AraSum columns: ['index', 'summary', 'article']


[2/4] EXTRACTIVE MODE (AraBERT)

Evaluating Extractive on 50 samples...

EVALUATION REPORT

Samples evaluated: 50
ROUGE-1:  0.2902
ROUGE-2:  0.1474
ROUGE-L:  0.2883
BLEU:     0.1983



[3/4] HYBRID MODE (AraBERT → AraT5)

Evaluating Hybrid on 50 samples...

EVALUATION REPORT

Samples evaluated: 50
ROUGE-1:  0.0000
ROUGE-2:  0.0000
ROUGE-L:  0.0000
BLEU:     0.0036


[4/4] ABSTRACTIVE MODE (AraT5)

Evaluating Abstractive on 50 samples...

EVALUATION REPORT

Samples evaluated: 50
ROUGE-1:  0.0333
ROUGE-2:  0.0000
ROUGE-L:  0.0333
BLEU:     0.0189


COMPARISON SUMMARY
Mode                      |  ROUGE-1 |  ROUGE-2 |  ROUGE-L |     BLEU
----------------------------------------------------------------------
Extractive (AraBERT)      |   0.2902 |   0.1474 |   0.2883 |   0.1983
Hybrid (AraBERT→AraT5)    |   0.0000 |   0.0000 |   0.0000 |   0.0036
Abstractive (AraT5)       |   0.0333 |   0.0000 |   0.0333 |   0.0189

Evaluation complete!
