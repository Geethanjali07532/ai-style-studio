"""
Extract Kaggle fashion dataset archive into data/raw/
"""
import os
import sys
import shutil
import zipfile
from pathlib import Path

# Ensure UTF-8 output encoding on Windows if supported
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DOWNLOADS_DIR = Path.home() / "Downloads"

ARCHIVE_PATH = DOWNLOADS_DIR / "archive.zip"
if not ARCHIVE_PATH.exists():
    # Check if user moved it to data/raw/
    alt_path = RAW_DATA_DIR / "archive.zip"
    if alt_path.exists():
        ARCHIVE_PATH = alt_path
    else:
        candidates = list(DOWNLOADS_DIR.glob("*fashion*.zip"))
        if candidates:
            ARCHIVE_PATH = candidates[0]
        else:
            print(f"[ERROR] Could not find archive.zip in {DOWNLOADS_DIR} or {RAW_DATA_DIR}")
            sys.exit(1)

print(f"[*] Found Kaggle Archive: {ARCHIVE_PATH} ({ARCHIVE_PATH.stat().st_size / (1024*1024):.1f} MB)")

# Step 1: Clean up any old dummy files in data/raw
print("[*] Preparing data/raw/ directory...")
raw_images = RAW_DATA_DIR / "images"
raw_styles = RAW_DATA_DIR / "styles.csv"
clean_meta = PROCESSED_DATA_DIR / "metadata_clean.csv"

if raw_styles.exists():
    raw_styles.unlink()
if clean_meta.exists():
    clean_meta.unlink()

# Remove old dummy images
if raw_images.exists():
    for f in raw_images.glob("100*.jpg"):
        try:
            f.unlink()
        except Exception:
            pass

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Step 2: Extract archive
print(f"[*] Extracting files to {RAW_DATA_DIR} ... (this takes about 20-30 seconds)")
with zipfile.ZipFile(ARCHIVE_PATH, "r") as zf:
    zf.extractall(RAW_DATA_DIR)

print("[OK] Extraction complete!")

# Check results
extracted_styles = RAW_DATA_DIR / "styles.csv"
if not extracted_styles.exists():
    # Check if inside a subfolder
    sub_styles = list(RAW_DATA_DIR.rglob("styles.csv"))
    if sub_styles:
        shutil.copy(sub_styles[0], extracted_styles)
        print(f"[*] Relocated styles.csv to {extracted_styles}")

images_dir = RAW_DATA_DIR / "images"
if not images_dir.exists():
    sub_images = [d for d in RAW_DATA_DIR.rglob("images") if d.is_dir()]
    if sub_images:
        images_dir = sub_images[0]

img_count = len(list(images_dir.glob("*.jpg")))
print(f"[*] Total fashion images found: {img_count:,}")
print("[SUCCESS] Dataset ready for ingestion!")
