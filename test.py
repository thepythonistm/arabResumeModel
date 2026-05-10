from src.pipeline import HybridArabicSummarizationPipeline

# Load hybrid pipeline
pipeline = HybridArabicSummarizationPipeline("./model")

# Test article
article = "الاقتصاد المصري يشهد تحسناً ملحوظاً في الفترة الأخيرة مع ارتفاع معدلات النمو وانخفاض معدلات البطالة. كما سجلت البورصة أعلى مستوياتها في تاريخها."

# Test all modes
result = pipeline.summarize(article, mode="both", top_n=3)

print("=" * 60)
print("EXTRACTIVE (AraBERT only):")
print("=" * 60)
print(result["extractive"])

print("\n" + "=" * 60)
print("ABSTRACTIVE (AraT5 on full text):")
print("=" * 60)
print(result["abstractive"])

print("\n" + "=" * 60)
print("HYBRID (AraBERT → AraT5):")
print("=" * 60)
print(result["hybrid"])