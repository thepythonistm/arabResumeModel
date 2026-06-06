"""
Download pre-trained models from Google Drive.
Run: python download_models.py
"""
import os
import sys

try:
    import gdown
except ImportError:
    print("Installing gdown...")
    os.system(f"{sys.executable} -m pip install gdown -q")
    import gdown


# ========== GOOGLE DRIVE FOLDER IDs ==========
EXTRACTIVE_FOLDER_ID = "1Ld4QwNejkEhx7AFBPD8at8v_xSJbee3I"
ABSTRACTIVE_FOLDER_ID = "1wqF4kYOZmQlEYiBN4z6GZ5XJ9pW5B57Q"
EXTRACTIVE_EASC_FOLDER_ID = "1DbjMUlo_HVcI4cMxaR6f5oFoaRSQ244t"


BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")

EXTRACTIVE_DIR = os.path.join(MODEL_DIR, "extractive")
EXTRACTIVE_EASC_DIR = os.path.join(MODEL_DIR, "extractive_easc")
ABSTRACTIVE_DIR = os.path.join(MODEL_DIR, "abstractive", "best")


def download_folder(folder_id, output_dir, desc):
    """Download a Google Drive folder using gdown."""
    os.makedirs(output_dir, exist_ok=True)

    has_model = any(
        f.endswith((".bin", ".safetensors"))
        for f in os.listdir(output_dir)
    ) if os.listdir(output_dir) else False

    if has_model:
        print(f"   {desc} already exists. Skipping.")
        return True

    print(f"\n   Downloading {desc}...")
    print(f"   From: https://drive.google.com/drive/folders/{folder_id}")
    print(f"   To:   {output_dir}")

    try:
        gdown.download_folder(
            id=folder_id,
            output=output_dir,
            quiet=False,
            use_cookies=False,
        )
        print(f"   {desc} downloaded successfully!")
        return True
    except Exception as e:
        print(f"   Error: {e}")
        return False


def main():
    print("=" * 55)
    print("Arabic Summarizer — Model Downloader")
    print("=" * 55)

    print("\nWhich extractive model do you want?")
    print("  1. Original (AraSum + WikiHow-Ar) — RECOMMENDED")
    print("  2. EASC Benchmark — for academic comparison")
    choice = input("Enter 1 or 2 (default: 1): ").strip() or "1"

    success = True

    success &= download_folder(ABSTRACTIVE_FOLDER_ID, ABSTRACTIVE_DIR, "Abstractive Model (AraT5)")

    if choice == "2":
        print("\n   [EASC Benchmark selected]")
        success &= download_folder(EXTRACTIVE_EASC_FOLDER_ID, EXTRACTIVE_EASC_DIR, "Extractive Model (EASC)")
        print(f"\n   Use: ArabicSummarizer('./model/extractive_easc', './model/abstractive/best')")
    else:
        success &= download_folder(EXTRACTIVE_FOLDER_ID, EXTRACTIVE_DIR, "Extractive Model (Original)")
        print(f"\n   Use: ArabicSummarizer('./model/extractive', './model/abstractive/best')")

    print("\n" + "=" * 55)
    if success:
        print("All models ready!")
        print(f"Models saved to: {MODEL_DIR}")
    else:
        print("Some downloads failed.")
        print("If gdown fails due to Drive quota, download manually and place in:")
        print(f"   {MODEL_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    main()