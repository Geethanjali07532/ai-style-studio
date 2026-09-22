"""
Training Script for Clothing Category Classifier.
Module 6: Clothing Category Classification.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
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
import tensorflow as tf


from src.config import (
    SPLITS_CSV,
    CLASSIFIER_MODEL_PATH,
    TARGET_IMG_SIZE,
    BATCH_SIZE,
    NUM_CLASSES,
    RANDOM_STATE,
)
from src.preprocessing import create_tf_dataset, create_stratified_splits
from src.classifier import (
    build_category_classifier,
    train_category_classifier,
    evaluate_classifier,
    ClothingClassifier,
)

GREEN = "\033[92m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def main():
    parser = argparse.ArgumentParser(description="Module 6: Train Clothing Category Classifier")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--subset-size", type=int, default=2000, help="Number of items to train on (0 for full 35k)")
    parser.add_argument("--backbone", type=str, default="mobilenet_v2", help="CNN Backbone: mobilenet_v2 or resnet50")
    args = parser.parse_args()

    print("\n" + "=" * 75)
    print("  AI Outfit Recommendation - Module 6: Clothing Category Classification")
    print("=" * 75 + "\n")

    # Step 1: Ensure dataset splits are available
    if not SPLITS_CSV.exists():
        print("[*] Generating stratified dataset splits first...")
        create_stratified_splits()

    df_splits = pd.read_csv(SPLITS_CSV)
    train_df = df_splits[df_splits["split"] == "train"]
    val_df = df_splits[df_splits["split"] == "val"]
    test_df = df_splits[df_splits["split"] == "test"]

    # If subset requested for rapid execution/CPU training
    if args.subset_size > 0 and len(train_df) > args.subset_size:
        print(f"[*] Subsetting training data to {args.subset_size} items for rapid CPU training...")

        def _sample_balanced(df_to_sample, target_count):
            n_per_cat = max(1, int(target_count / NUM_CLASSES))
            indices = []
            rng = np.random.RandomState(RANDOM_STATE)
            for cat in df_to_sample["canonical_category"].unique():
                cat_idx = df_to_sample[df_to_sample["canonical_category"] == cat].index
                chosen = rng.choice(cat_idx, size=min(len(cat_idx), n_per_cat), replace=False)
                indices.extend(chosen)
            return df_to_sample.loc[indices].copy()

        train_df = _sample_balanced(train_df, args.subset_size)
        val_df = _sample_balanced(val_df, int(args.subset_size * 0.15))
        test_df = _sample_balanced(test_df, int(args.subset_size * 0.15))


    print(f"[*] Training Items   : {len(train_df):,}")
    print(f"[*] Validation Items : {len(val_df):,}")
    print(f"[*] Test Items       : {len(test_df):,}")

    # Step 2: Build high-performance tf.data streaming pipelines
    train_ds = create_tf_dataset(train_df, batch_size=args.batch_size, is_training=True, augment=True)
    val_ds = create_tf_dataset(val_df, batch_size=args.batch_size, is_training=False, augment=False)
    test_ds = create_tf_dataset(test_df, batch_size=args.batch_size, is_training=False, augment=False)

    # Step 3: Build Transfer Learning Model
    print(f"\n[*] Building {args.backbone} Transfer Learning Classifier (ImageNet weights)...")
    model = build_category_classifier(
        backbone=args.backbone,
        input_shape=(TARGET_IMG_SIZE[0], TARGET_IMG_SIZE[1], 3),
        num_classes=NUM_CLASSES,
        freeze_backbone=True,
        learning_rate=1e-3,
    )
    model.summary(line_length=80)

    # Step 4: Train with callbacks
    history = train_category_classifier(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=args.epochs,
        checkpoint_path=CLASSIFIER_MODEL_PATH,
    )

    # Step 5: Evaluate on Test Set
    print("\n[*] Evaluating Best Model on Test Dataset...")
    best_model = tf.keras.models.load_model(str(CLASSIFIER_MODEL_PATH))
    results = evaluate_classifier(best_model, test_ds, test_df)

    print(f"\n{BOLD}Test Set Performance Summary:{RESET}")
    print(f"  * Test Loss           : {results['test_loss']:.4f}")
    print(f"  * Test Accuracy       : {GREEN}{results['test_accuracy']*100:.2f}%{RESET}")
    print(f"  * Top-3 Accuracy      : {GREEN}{results['test_top3_accuracy']*100:.2f}%{RESET}")
    print(f"  * Confusion Matrix    : {results['confusion_matrix_path']}")

    # Step 6: Test Real Inference on Sample Images
    print(f"\n{BOLD}[Module 6 Smoke Test] Live Inference with ClothingClassifier:{RESET}")
    classifier = ClothingClassifier(model_path=CLASSIFIER_MODEL_PATH)
    sample_rows = test_df.sample(min(3, len(test_df)), random_state=RANDOM_STATE)

    for _, row in sample_rows.iterrows():
        pred = classifier.predict(row["image_path"], top_k=3)
        actual = row["canonical_category"]
        match_str = f"{GREEN}[MATCH]{RESET}" if pred["top_category"] == actual else "[DIFF]"
        formatted_top3 = [f"{x['category']}: {x['probability']*100:.1f}%" for x in pred['top_k']]
        print(f"  * Actual: {actual:<12} | Predicted: {pred['top_category']:<12} (Conf: {pred['confidence']*100:.1f}%) {match_str}")
        print(f"    Top-3: {formatted_top3}")


    print(f"\n{BOLD}{GREEN}[SUCCESS] Module 6 (Clothing Category Classification) Complete!{RESET}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
