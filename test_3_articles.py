#!/usr/bin/env python3
"""
Test the hybrid summarizer on 3 Arabic articles.
Run: python test_3_articles.py
"""
from src.hybrid import ArabicSummarizer

# 3 sample articles
articles = [
    # Article 1: Economics
    """القاهرة - أعلن البنك المركزي المصري اليوم عن قراره بتثبيت أسعار الفائدة على 
    الإيداع والإقراض لليلة واحدة عند 18.25% و19.25% على التوالي. 
    وجاء القرار خلال اجتماع لجنة السياسة النقدية الذي عقد الخميس الماضي. 
    وأوضح البنك في بيانه أن هذا القرار يأتي في إطار سعيه لتحقيق استقرار الأسعار 
    والحد من التضخم الذي وصل إلى 33% في أبريل الماضي. 
    وأضاف أن المؤشرات الاقتصادية تظهر بوادر تحسن في معدلات النمو.""",

    # Article 2: Health
    """أعلنت وزارة الصحة السعودية عن تسجيل 150 إصابة جديدة بفيروس كورونا المستجد 
    خلال الـ24 ساعة الماضية. وقالت الوزارة إن الحالات الجديدة توزعت بين عدة مدن 
    مع تركز معظمها في الرياض وجدة. وأوضحت أنه تم تقديم اللقاحات لأكثر من 30 مليون 
    مواطن ومقيم حتى الآن. ودعت الوزارة المواطنين إلى الالتزام بالإجراءات الاحترازية 
    والحفاظ على التباعد الاجتماعي.""",

    # Article 3: Technology
    """كشفت شركة أبل الأمريكية عن هاتفها الجديد آيفون 16 بمواصفات تقنية متطورة. 
    وقالت الشركة إن الهاتف يأتي بمعالج A18 Pro الأسرع من أي جيل سابق. 
    ويتميز الهاتف بكاميرا خلفية ثلاثية بدقة 48 ميجابكسل مع تحسينات في التصوير الليلي. 
    وسيكون متوفراً في الأسواق العالمية بداية من الشهر المقبل بأسعار تبدأ من 999 دولاراً.""",
]

domains = ["اقتصاد", "صحة", "تقنية"]

# Initialize
summarizer = ArabicSummarizer(
    extractive_model_path="./model/extractive",
    abstractive_model_path="./model/abstractive/best",
)

# Batch summarize
print("=" * 60)
print("BATCH SUMMARIZATION — 3 ARTICLES")
print("=" * 60)

results = summarizer.summarize_batch(articles, domains=domains)

for i, (article, result) in enumerate(zip(articles, results), 1):
    print(f"\n{'─' * 60}")
    print(f"📄 ARTICLE {i} [{domains[i-1]}]")
    print(f"{'─' * 60}")
    print(f"Input: {article[:100]}...")
    print(f"\n📝 SUMMARY:")
    print(result["summary"])

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)