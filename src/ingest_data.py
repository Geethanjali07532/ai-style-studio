"""
Dataset Ingestion, Auto-Extraction, and Validation Script.
Module 3: Fashion Dataset Collection & Understanding.
"""
from __future__ import annotations
import argparse
import os
import sys
import zipfile
from pathlib import Path

# Add project root to sys.path to allow direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output encoding on Windows if supported
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_STYLES_CSV,
    RAW_IMAGES_DIR,
    CLEANED_METADATA_CSV,
    CLOTHING_CATEGORIES,
    CATEGORY_TO_OUTFIT_PART,
)
from src.dataset import FashionDataset


# ANSI terminal colors for formatted output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def auto_extract_zips(target_dir: Path) -> bool:
    """Scans target_dir for .zip files and extracts them."""
    zip_files = list(target_dir.glob("*.zip"))
    if not zip_files:
        return False

    print(f"\n{BOLD}{CYAN}[*] Found {len(zip_files)} ZIP archive(s) in {target_dir}:{RESET}")
    for zpath in zip_files:
        print(f"  * Extracting archive: {YELLOW}{zpath.name}{RESET} ({zpath.stat().st_size / (1024*1024):.1f} MB)...")
        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(target_dir)
            print(f"  * Extraction complete: {GREEN}[OK]{RESET}")
        except Exception as e:
            print(f"  * {RED}Error extracting {zpath.name}: {e}{RESET}")
    return True


def create_demo_starter_dataset(num_items_per_cat: int = 4):
    """
    Creates a high-quality starter sample catalog in data/raw/
    with synthetic fashion garment silhouettes and clean styles.csv.
    Enables immediate testing of all CV and outfit matching modules.
    """
    print(f"\n{BOLD}{CYAN}[*] Generating starter fashion catalog in data/raw/...{RESET}")

    RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Color definitions (RGB)
    COLOR_PALETTE = {
        "Black": (30, 30, 30),
        "White": (240, 240, 240),
        "Navy Blue": (20, 35, 75),
        "Burgundy": (110, 20, 40),
        "Olive Green": (85, 107, 47),
        "Beige": (220, 200, 170),
        "Grey": (128, 128, 128),
        "Denim Blue": (45, 85, 140),
    }
    color_names = list(COLOR_PALETTE.keys())

    items = []
    current_id = 10001

    for category in CLOTHING_CATEGORIES:
        for i in range(num_items_per_cat):
            item_id = str(current_id)
            color_name = color_names[(current_id) % len(color_names)]
            color_rgb = COLOR_PALETTE[color_name]
            gender = "Men" if i % 2 == 0 else "Women"
            usage = "Casual" if i < 2 else "Formal"

            # Create synthetic fashion visual
            img = Image.new("RGB", (224, 224), (248, 248, 250))
            draw = ImageDraw.Draw(img)

            # Draw background card
            draw.rectangle([10, 10, 214, 214], fill=(255, 255, 255), outline=(225, 225, 230), width=2)

            # Draw category-specific garment silhouette
            if category in ["T-Shirt", "Shirt"]:
                # Torso & sleeves
                draw.polygon([(60, 50), (164, 50), (200, 100), (170, 115), (150, 85), (150, 190), (74, 190), (74, 85), (54, 115), (24, 100)], fill=color_rgb)
                # Collar / neckline
                draw.arc([92, 40, 132, 70], start=0, end=180, fill=(248, 248, 250), width=4)
            elif category in ["Jeans", "Trousers"]:
                # Waistband to cuffs
                draw.polygon([(65, 40), (159, 40), (155, 195), (120, 195), (112, 90), (104, 195), (69, 195)], fill=color_rgb)
                # Pocket line
                draw.line([(75, 60), (100, 60)], fill=(200, 200, 200), width=2)
            elif category == "Dress":
                # Bodice and flared skirt
                draw.polygon([(85, 40), (139, 40), (130, 95), (175, 195), (49, 195), (94, 95)], fill=color_rgb)
            elif category == "Skirt":
                # High waist flared skirt
                draw.polygon([(80, 60), (144, 60), (175, 185), (49, 185)], fill=color_rgb)
            elif category == "Jacket":
                # Outer coat structure
                draw.rectangle([55, 45, 169, 190], fill=color_rgb)
                draw.polygon([(112, 45), (90, 120), (112, 190), (134, 120)], fill=(color_rgb[0]//2, color_rgb[1]//2, color_rgb[2]//2))
            elif category in ["Shoes", "Sneakers"]:
                # Shoe profile
                draw.polygon([(35, 140), (140, 135), (195, 155), (190, 180), (30, 180)], fill=color_rgb)
                # White sole
                draw.rectangle([30, 175, 190, 185], fill=(230, 230, 230))
            else:  # Accessories
                # Circular watch / accessory
                draw.ellipse([67, 67, 157, 157], fill=color_rgb, outline=(200, 180, 50), width=4)
                draw.ellipse([82, 82, 142, 142], fill=(250, 250, 250))

            # Save generated image
            img_path = RAW_IMAGES_DIR / f"{item_id}.jpg"
            img.save(img_path, "JPEG", quality=92)

            display_name = f"{color_name} {usage} {category}"
            items.append({
                "id": item_id,
                "gender": gender,
                "masterCategory": "Apparel" if category not in ["Shoes", "Sneakers", "Accessories"] else ("Footwear" if "Shoe" in category or category == "Sneakers" else "Accessories"),
                "subCategory": category,
                "articleType": category,
                "baseColour": color_name,
                "season": "All-Season",
                "year": 2024,
                "usage": usage,
                "productDisplayName": display_name,
            })
            current_id += 1

    # Save styles.csv
    df = pd.DataFrame(items)
    df.to_csv(RAW_STYLES_CSV, index=False)
    print(f"  * Generated {len(items)} catalog items across {len(CLOTHING_CATEGORIES)} categories.")
    print(f"  * Images saved to: {RAW_IMAGES_DIR}")
    print(f"  * Metadata saved to: {RAW_STYLES_CSV}\n")


def print_catalog_report(stats: dict):
    """Renders a rich CLI status report of the dataset."""
    print(f"\n{CYAN}{'=' * 75}{RESET}")
    print(f"{BOLD}{CYAN}  AI Outfit Recommendation & Style Matching System - Module 3 Report{RESET}")
    print(f"{CYAN}  Fashion Dataset Catalog & Ingestion Diagnostic{RESET}")
    print(f"{CYAN}{'=' * 75}{RESET}\n")

    print(f"  * {BOLD}Total Catalog Items{RESET}      : {GREEN}{stats.get('total_items', 0):,}{RESET}")
    print(f"  * {BOLD}Valid Images Found on Disk{RESET}: {GREEN}{stats.get('images_found', 0):,}{RESET}")

    categories = stats.get("categories", {})
    if categories:
        print(f"\n{BOLD}  Category Breakdown:{RESET}")
        for cat, count in list(categories.items())[:12]:
            bar = "█" * min(25, max(1, int(count / max(categories.values()) * 25)))
            print(f"    - {cat:<16}: {count:>6} items  {CYAN}{bar}{RESET}")

    parts = stats.get("outfit_parts", {})
    if parts:
        print(f"\n{BOLD}  Outfit Part Distribution (Top/Bottom/Shoes/Accessories):{RESET}")
        for part, count in parts.items():
            print(f"    - {part:<14}: {count:>6} items")

    genders = stats.get("gender_distribution", {})
    if genders:
        print(f"\n{BOLD}  Gender Segments:{RESET}")
        for g, count in genders.items():
            print(f"    - {g:<14}: {count:>6} items")

    print(f"\n{CYAN}{'=' * 75}{RESET}")


def main():
    parser = argparse.ArgumentParser(description="Module 3: Fashion Dataset Ingestion and Validation")
    parser.add_argument("--create-demo-sample", action="store_true", help="Generate starter fashion items if no dataset is present")
    parser.add_argument("--zip-path", type=str, default=None, help="Explicit path to downloaded Kaggle ZIP archive")
    args = parser.parse_args()

    # Step 1: Auto-extract ZIP if provided or found in data/raw/
    if args.zip_path and Path(args.zip_path).exists():
        with zipfile.ZipFile(args.zip_path, "r") as zf:
            zf.extractall(RAW_DATA_DIR)
        print(f"{GREEN}[OK] Extracted {args.zip_path} into {RAW_DATA_DIR}{RESET}")
    else:
        auto_extract_zips(RAW_DATA_DIR)

    # Step 2: Initialize dataset manager
    dataset = FashionDataset()

    # If dataset is empty and flag requested or nothing in raw, give options
    if not dataset.is_loaded():
        if args.create_demo_sample:
            create_demo_starter_dataset()
            dataset = FashionDataset()
        else:
            print(f"\n{YELLOW}[NOTICE] No fashion dataset detected in data/raw/.{RESET}")
            print(f"To load your Kaggle dataset:")
            print(f"  1. Place downloaded zip file (e.g. fashion-product-images-small.zip) in {RAW_DATA_DIR}")
            print(f"     OR extract it so {RAW_STYLES_CSV} and {RAW_IMAGES_DIR} exist.")
            print(f"  2. Re-run: {BOLD}python src/ingest_data.py{RESET}")
            print(f"\nAlternatively, run with {BOLD}--create-demo-sample{RESET} to generate a starter visual catalog.")
            return 1

    # Step 3: Validate and save cleaned metadata
    saved_path = dataset.save_cleaned_metadata()
    stats = dataset.get_summary_stats()
    print_catalog_report(stats)

    # Step 4: Sample a mock outfit to verify recommendation readiness
    sample_outfit = dataset.sample_random_outfit()
    print(f"{BOLD}[Module 3 Test] Candidate Outfit Sample Generator:{RESET}")
    for part, item in sample_outfit.items():
        if item:
            print(f"  * {part.capitalize():<10}: ID {item.get('id')} - {item.get('productDisplayName', 'Item')} ({item.get('canonical_category')})")
        else:
            print(f"  * {part.capitalize():<10}: None available")

    print(f"\n{BOLD}{GREEN}[SUCCESS] Module 3 (Fashion Dataset Collection & Understanding) Ready!{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
