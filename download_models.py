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


# ========== PASTE YOUR IDs HERE ==========
EXTRACTIVE_FOLDER_ID = "1Ld4QwNejkEhx7AFBPD8at8v_xSJbee3I"      
ABSTRACTIVE_FOLDER_ID = "1wqF4kYOZmQlEYiBN4z6GZ5XJ9pW5B57Q"    


BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")

EXTRACTIVE_DIR = os.path.join(MODEL_DIR, "extractive", "best")
ABSTRACTIVE_DIR = os.path.join(MODEL_DIR, "abstractive", "best")


def download_folder(folder_id, output_dir, desc):
    """Download a Google Drive folder using gdown."""
    os.makedirs(output_dir, exist_ok=True)
    
    pytorch_bin = os.path.join(output_dir, "pytorch_model.bin")
    safetensors = os.path.join(output_dir, "model.safetensors")
    
    if os.path.exists(pytorch_bin) or os.path.exists(safetensors):
        print(f"✅ {desc} already exists at {output_dir}")
        return True
    
    if "YOUR_" in folder_id:
        print(f"\n❌ ERROR: Replace placeholder ID for {desc}")
        print(f"   Edit EXTRACTIVE_FOLDER_ID or ABSTRACTIVE_FOLDER_ID in this file")
        return False
    
    print(f"\n📥 Downloading {desc}...")
    print(f"   From: https://drive.google.com/drive/folders/{folder_id}")
    print(f"   To:   {output_dir}")
    
    try:
        gdown.download_folder(
            id=folder_id,
            output=output_dir,
            quiet=False,
            use_cookies=False
        )
        print(f"✅ {desc} downloaded")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def main():
    print("=" * 50)
    print("Arabic Summarization — Model Downloader")
    print("=" * 50)
    
    if "YOUR_" in EXTRACTIVE_FOLDER_ID or "YOUR_" in ABSTRACTIVE_FOLDER_ID:
        print("\n⚠️  WARNING: You haven't set your Google Drive folder IDs yet!")
        print("   Open download_models.py and paste your IDs at the top.\n")
        return
    
    success = True
    success &= download_folder(EXTRACTIVE_FOLDER_ID, EXTRACTIVE_DIR, "Extractive Model (AraBERT)")
    success &= download_folder(ABSTRACTIVE_FOLDER_ID, ABSTRACTIVE_DIR, "Abstractive Model (AraT5)")
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All models ready!")
        print(f"   Extractive:  {EXTRACTIVE_DIR}")
        print(f"   Abstractive: {ABSTRACTIVE_DIR}")
        print("\n   Run: python src/hybrid.py")
    else:
        print("❌ Some downloads failed. Check errors above.")
    print("=" * 50)


if __name__ == "__main__":
    main()