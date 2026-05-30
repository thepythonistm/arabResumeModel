"""Test script for the Arabic hybrid summarizer."""
from src.hybrid import ArabicSummarizer

text = """
القاهرة - أعلن البنك المركزي المصري اليوم عن قراره بتثبيت أسعار الفائدة على 
الإيداع والإقراض لليلة واحدة عند 18.25% و19.25% على التوالي. 
وجاء القرار خلال اجتماع لجنة السياسة النقدية الذي عقد الخميس الماضي. 
وأوضح البنك في بيانه أن هذا القرار يأتي في إطار سعيه لتحقيق استقرار الأسعار 
والحد من التضخم الذي وصل إلى 33% في أبريل الماضي. 
وأضاف أن المؤشرات الاقتصادية تظهر بوادر تحسن في معدلات النمو.
"""

summarizer = ArabicSummarizer(
    extractive_model_path="./model/extractive",
    abstractive_model_path="./model/abstractive/best",
)

result = summarizer.summarize(text, domain="news", return_full=True)

print("\n" + "=" * 50)
for key, value in result.items():
    print(f"\n{key.upper()}:")
    if isinstance(value, list):
        print(value)
    else:
        print(value)