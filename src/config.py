"""
Configuration and centralized paths for AI Outfit Recommendation System.
"""
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
APP_DIR = PROJECT_ROOT / "app"
TESTS_DIR = PROJECT_ROOT / "tests"
REPORTS_DIR = PROJECT_ROOT / "reports"
EDA_FIGURES_DIR = REPORTS_DIR / "eda_figures"

# Specific File Paths
RAW_STYLES_CSV = RAW_DATA_DIR / "styles.csv"
RAW_IMAGES_DIR = RAW_DATA_DIR / "images"
CLEANED_METADATA_CSV = PROCESSED_DATA_DIR / "metadata_clean.csv"
SPLITS_CSV = PROCESSED_DATA_DIR / "dataset_splits.csv"
EDA_SUMMARY_JSON = PROCESSED_DATA_DIR / "eda_summary.json"
CLASSIFIER_MODEL_PATH = MODELS_DIR / "category_classifier.keras"

CONFUSION_MATRIX_PNG = EDA_FIGURES_DIR / "confusion_matrix.png"
EMBEDDINGS_NPY = EMBEDDINGS_DIR / "fashion_embeddings.npy"
EMBEDDINGS_INDEX_CSV = EMBEDDINGS_DIR / "embedding_index.csv"



# Ensure essential directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR, MODELS_DIR, NOTEBOOKS_DIR, APP_DIR, TESTS_DIR, REPORTS_DIR, EDA_FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# Image Preprocessing & Model Constants
TARGET_IMG_SIZE = (224, 224)
DEFAULT_IMAGE_SIZE = TARGET_IMG_SIZE
COLOR_CHANNELS = 3
BATCH_SIZE = 32
RANDOM_STATE = 42

# Feature Extractor Settings
FEATURE_BACKBONE = "mobilenet_v2"
EMBEDDING_DIM = 1280

# Normalization Modes
NORMALIZATION_MODES = ["scale_0_1", "tf_minus1_to_1", "imagenet"]
DEFAULT_NORMALIZATION = "scale_0_1"


# Dataset Splits
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

# Target Canonical Clothing Categories
CLOTHING_CATEGORIES = [
    "T-Shirt",
    "Shirt",
    "Jeans",
    "Trousers",
    "Dress",
    "Skirt",
    "Jacket",
    "Shoes",
    "Sneakers",
    "Accessories",
]

NUM_CLASSES = len(CLOTHING_CATEGORIES)
CATEGORY_TO_IDX = {cat: idx for idx, cat in enumerate(CLOTHING_CATEGORIES)}
IDX_TO_CATEGORY = {idx: cat for idx, cat in enumerate(CLOTHING_CATEGORIES)}

# High-Level Outfit Component Types
OUTFIT_PARTS = ["top", "bottom", "shoes", "outerwear", "accessory"]


# Mapping from canonical category to outfit part
CATEGORY_TO_OUTFIT_PART = {
    "T-Shirt": "top",
    "Shirt": "top",
    "Jeans": "bottom",
    "Trousers": "bottom",
    "Skirt": "bottom",
    "Dress": "top",       # Standalone or top-level garment
    "Jacket": "outerwear",
    "Shoes": "shoes",
    "Sneakers": "shoes",
    "Accessories": "accessory",
}

# Mapping from Kaggle Fashion Dataset articleType to canonical categories
KAGGLE_ARTICLE_MAPPING = {
    # Tops
    "Tshirts": "T-Shirt",
    "T-shirts": "T-Shirt",
    "Tops": "T-Shirt",
    "Shirts": "Shirt",
    "Formal Shirts": "Shirt",
    "Casual Shirts": "Shirt",
    "Tunics": "Shirt",
    "Kurtas": "Shirt",
    "Sweatshirts": "T-Shirt",
    "Sweaters": "T-Shirt",
    
    # Bottoms
    "Jeans": "Jeans",
    "Trousers": "Trousers",
    "Track Pants": "Trousers",
    "Capris": "Trousers",
    "Shorts": "Trousers",
    "Skirts": "Skirt",
    "Leggings": "Trousers",
    
    # Dresses
    "Dresses": "Dress",
    "Jumpsuit": "Dress",
    
    # Outerwear
    "Jackets": "Jacket",
    "Blazers": "Jacket",
    "Coats": "Jacket",
    "Shrug": "Jacket",
    "Rain Jacket": "Jacket",
    
    # Shoes & Footwear
    "Casual Shoes": "Shoes",
    "Formal Shoes": "Shoes",
    "Flats": "Shoes",
    "Heels": "Shoes",
    "Sandals": "Shoes",
    "Flip Flops": "Shoes",
    "Sports Shoes": "Sneakers",
    
    # Accessories
    "Watches": "Accessories",
    "Belts": "Accessories",
    "Sunglasses": "Accessories",
    "Bags": "Accessories",
    "Handbags": "Accessories",
    "Backpacks": "Accessories",
    "Wallets": "Accessories",
    "Clutches": "Accessories",
    "Ties": "Accessories",
    "Scarves": "Accessories",
    "Caps": "Accessories",
    "Socks": "Accessories",
    "Jewellery": "Accessories",
}
