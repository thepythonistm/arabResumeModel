from src.pipeline import HybridArabicSummarizationPipeline

text = """
القاهرة - أعلن البنك المركزي المصري اليوم عن قراره بتثبيت أسعار الفائدة على 
الإيداع والإقراض لليلة واحدة عند 18.25% و19.25% على التوالي. 
وجاء القرار خلال اجتماع لجنة السياسة النقدية الذي عقد الخميس الماضي. 
وأوضح البنك في بيانه أن هذا القرار يأتي في إطار سعيه لتحقيق استقرار الأسعار 
والحد من التضخم الذي وصل إلى 33% في أبريل الماضي. 
وأضاف أن المؤشرات الاقتصادية تظهر بوادر تحسن في معدلات النمو.
"""

pipeline = HybridArabicSummarizationPipeline("./model")
result = pipeline.summarize(text, mode="both", top_n=3, debug=True)

print("\n" + "="*50)
for key, value in result.items():
    print(f"\n{key.upper()}:")
    print(value)