# AI Outfit Recommendation & Style Matching System
> **Computer Vision & Deep Learning Based Fashion Recommendation Engine**

---

## 📌 Project Overview
The **AI Outfit Recommendation & Style Matching System** is an intelligent fashion assistant that uses computer vision and deep learning to recommend matching clothing items and complete outfit combinations. Given an uploaded or selected clothing image, the system analyzes its visual characteristics — color, pattern, texture, category, and overall style — and suggests compatible items that form a coherent, well-matched outfit.

---

## 🛠️ Technologies & Stack

| Technology | Purpose |
| :--- | :--- |
| **Python 3.13** | Core programming language for the entire project |
| **NumPy** | Numerical computation and array/vector operations |
| **Pandas** | Dataset handling, labeling, and organization |
| **OpenCV** | Image loading, processing, and computer vision operations |
| **PIL / Pillow** | Image manipulation and format conversion |
| **Matplotlib / Seaborn** | Data visualization and exploratory analysis |
| **Scikit-learn** | Similarity metrics, clustering, and classical ML utilities |
| **TensorFlow / Keras** | CNN-based feature extraction and deep learning models |
| **Streamlit** | Interactive outfit recommendation web application |

---

## 📁 Project Directory Structure

```text
├── app/                  # Streamlit web application frontend (Module 14)
├── data/
│   ├── raw/              # Raw fashion dataset images (Module 3)
│   ├── processed/        # Resized and normalized images (Module 4)
│   └── embeddings/       # Extracted feature embeddings (Module 8)
├── models/               # Saved CNN models, checkpoints, and weights (Module 6)
├── notebooks/            # Jupyter notebooks for experiments & EDA (Module 5)
├── src/                  # Core Python modules and packages
│   ├── __init__.py
│   └── config.py         # Central configuration, paths, and constants
├── tests/                # Unit tests
├── verify_env.py         # Automated environment diagnostic and verification script
├── requirements.txt      # Project dependencies
├── .gitignore            # Git ignore file
└── README.md             # Project documentation
```

---

## 🗺️ 15-Module Roadmap

- [x] **Module 1: Project Overview & Fashion Recommendation Architecture** — Architecture design, scope, and pipeline definition.
- [x] **Module 2: Environment Setup & Computer Vision Libraries** — Python runtime, dependencies, directory structure, and diagnostics.
- [x] **Module 3: Fashion Dataset Collection & Understanding** — Sourcing, automated ingestion, category normalization, and catalog validation.
- [x] **Module 4: Fashion Image Preprocessing** — Aspect-ratio letterboxing, normalization protocols, data augmentation, stratified splits, and tf.data streaming.
- [x] **Module 5: Fashion Image Exploratory Analysis** — Category distributions, K-Means dominant color extraction, and fashion color harmony matching.
- [x] **Module 6: Clothing Category Classification** — CNN-based classification with MobileNetV2, callbacks, confusion matrix, and inference engine.
- [x] **Module 7: Fashion Feature Extraction** — Deep visual feature extraction using pretrained MobileNetV2, L2 normalization, and intermediate feature map visualization.
- [x] **Module 8: Image Embedding Generation** — High-dimensional vector representations, L2-normalized catalog matrix (.npy), and sub-millisecond retrieval engine.
- [x] **Module 9: Clothing Similarity & Style Matching** — Vector cosine similarity, fashion color harmony theory, occasion alignment, and composite outfit compatibility scoring.
- [x] **Module 10: Outfit Recommendation Engine** — Multi-item outfit generation (top, bottom, shoes, accessories), beam search assembly, item substitution, and global cohesion scoring.
- [x] **Module 11: User Preference & Personalization** — User style profiles, demographic constraints, persona presets, item affinity functions, and adaptive personalized outfit re-ranking.
- [x] **Module 12: Recommendation Model Evaluation** — Precision@K, Recall@K, Hit Rate@K, Mean Reciprocal Rank (MRR), Intra-List Diversity (ILD), and Catalog Coverage benchmarking.
- [x] **Module 14: AI Outfit Recommendation Application** — Interactive editorial luxury fashion web application (AI Style Studio) featuring high-resolution Lanczos image upsampling, Style Studio, Occasion Lookbook, Similar Pieces, and How It Works.
- [x] **Module 15: Final AI Fashion Recommendation Capstone Project** — Complete end-to-end integration, 53 automated unit tests, and comprehensive verification.

---

## 🚀 Getting Started

### 1. Check Python Environment
Ensure Python 3.10+ is installed:
```powershell
python --version
```

### 2. Install Dependencies
Install all required libraries:
```powershell
pip install -r requirements.txt
```

### 3. Verify Environment Setup
Run the diagnostic script to test all computer vision and deep learning packages:
```powershell
python verify_env.py
```

### 4. Fashion Dataset Ingestion (Module 3)
To load your dataset (e.g. Kaggle [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)):
- Drop the `.zip` file into `data/raw/`, or extract it so `styles.csv` and `images/` exist in `data/raw/`.
- Run the automated dataset ingestor:
```powershell
python src/ingest_data.py
```
- Or generate a starter synthetic catalog for instant testing:
```powershell
python src/ingest_data.py --create-demo-sample
```
- Run unit tests:
```powershell
python -m unittest tests/test_dataset.py
```

### 5. Fashion Image Preprocessing & Splitting (Module 4)
- Generate stratified Train (80%), Val (10%), and Test (10%) splits:
```powershell
python -c "from src.preprocessing import create_stratified_splits; create_stratified_splits()"
```
- Run preprocessing unit tests:
```powershell
python -m unittest tests/test_preprocessing.py
```

### 6. Exploratory Data Analysis & Color Analytics (Module 5)
- Run automated visual EDA and generate report figures:
```powershell
python src/eda_analysis.py --sample-size 150
```
- Run color analytics unit tests:
```powershell
python -m unittest tests/test_eda.py
```

### 7. Clothing Category Classification (Module 6)
- Train the MobileNetV2 category classifier:
```powershell
python src/train_classifier.py --subset-size 2000 --epochs 3 --batch-size 32
```
- Run classifier unit tests:
```powershell
python -m unittest tests/test_classifier.py
```

### 8. Fashion Feature Extraction (Module 7)
- Run feature extraction smoke test:
```powershell
python -c "from src.feature_extractor import FashionFeatureExtractor; ext = FashionFeatureExtractor(); print('Embedding Dim:', ext.embedding_dim)"
```
- Run feature extraction unit tests:
```powershell
python -m unittest tests/test_feature_extractor.py
```

### 9. Image Embedding Generation & Vector Indexing (Module 8)
- Generate catalog visual embeddings and build index:
```powershell
python src/generate_embeddings.py --limit 2000 --batch-size 64
```
- Run embedding manager unit tests:
```powershell
python -m unittest tests/test_embeddings.py
```

### 10. Clothing Similarity & Style Matching (Module 9)
- Run style matching and outfit compatibility unit tests:
```powershell
python -m unittest tests/test_style_matcher.py
```

### 11. Multi-Item Outfit Recommendation (Module 10)
- Run outfit recommender unit tests:
```powershell
python -m unittest tests/test_recommender.py
```

### 12. User Preference & Personalization (Module 11)
- Run personalization unit tests:
```powershell
python -m unittest tests/test_personalization.py
```

### 13. Recommendation Evaluation & Benchmarking (Module 12)
- Run evaluation benchmark across real catalog:
```powershell
python src/run_evaluation.py --sim-queries 60 --outfit-queries 30
```
- Run evaluation unit tests:
```powershell
python -m unittest tests/test_evaluation.py
```

### 14. Recommendation Optimization & Threshold Tuning (Module 13)
- Run threshold tuning and latency optimization benchmark:
```powershell
python src/run_optimization.py --eval-pairs 200 --benchmark-queries 20
```
- Run optimization unit tests:
```powershell
python -m unittest tests/test_optimizer.py
```

### 15. Launch Interactive Web Application (Module 14 & 15)
Launch the Streamlit fashion stylist web application:
```powershell
streamlit run app/main.py
```
Or double-click `run_app.bat` in the project root.

Open your browser at **http://localhost:8501** to experience **AI Style Studio**:
- ✦ **Style Studio**: 7-step fashion journey: Start with one piece (Upload or Catalog) → AI Style Analysis → Visual Details → Similar Pieces → What Goes With It? (Complementary categories) → Your Looks (Complete outfits) → Personalized For You.
- ✦ **Occasion Looks**: Curated collections for Casual, College, Office, Party, and Formal.
- ✦ **Similar Pieces**: Visual similarity search powered by MobileNetV2 cosine similarity.
- ✦ **How It Works**: 9-step pipeline architecture diagram, verified system metrics, and classification heatmap.

Run complete test suite across all modules (53 tests):
```powershell
python -m unittest discover -s tests -p "test_*.py"
```





