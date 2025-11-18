# 🛒 Intelligent Product Price Prediction Engine

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![LightGBM](https://img.shields.io/badge/LightGBM-Framework-green)]()
[![Status](https://img.shields.io/badge/Status-Completed-success)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()

A modular end-to-end ML pipeline that predicts product prices from noisy, multi-modal catalog data (text descriptions, pack quantities, and brand names). Combines regex-based preprocessing, TF-IDF text features, OOF target encoding, and LightGBM models optimized with SMAPE.

## Table of contents
- [Highlights](#highlights)
- [Technical summary](#technical-summary)
- [Getting started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Install](#install)
  - [Run training](#run-training)
  - [Run demo app](#run-demo-app)
- [Repository structure](#repository-structure)
- [Evaluation & results](#evaluation--results)
- [Key implementation notes](#key-implementation-notes)
- [License](#license)
- [Contact](#contact)

## Highlights
- Handles unstructured product text and extracts structured attributes (e.g., volumes, pack sizes).
- Uses TF-IDF (word & char n-grams) for high-dimensional text features.
- Employs Out-of-Fold (OOF) target encoding for high-cardinality categorical features (Brand).
- Trains LightGBM with log1p target transform and optimizes SMAPE to balance errors across price ranges.

## Technical summary
- Language: Python 3.8+
- Model: LightGBM (gradient boosting)
- Feature engineering: Regex entity extraction, TF-IDF, OOF mean encoding
- Validation: Stratified K-Fold (5 folds)
- Metric: SMAPE (primary), MAE reported

## Getting started

### Prerequisites
- Python 3.8+
- pip

### Install
Clone and install dependencies:
```bash
git clone https://github.com/yourusername/price-prediction-engine.git
cd price-prediction-engine
pip install -r requirements.txt
```

### Run training
This runs the full pipeline: feature generation, training (5-fold LightGBM), and artifact saving to models/.
```bash
python main.py
```

### Run demo app (Streamlit)
Launch an interactive demo:
```bash
streamlit run app.py
```

## Repository structure
```
price_prediction_project/
├── data/                    # Raw and processed datasets (gitignored)
├── models/                  # Saved models (.txt) and vectorizers (.pkl)
├── notebooks/               # EDA and experiments
├── src/                     # Source code modules
│   ├── config.py            # Configs & hyperparams
│   ├── preprocessing.py     # Regex and text cleaning
│   ├── features.py          # TF-IDF, OOF encoding, feature gen
│   ├── training.py          # LightGBM training loop
│   └── utils.py             # Metrics and helpers
├── app.py                   # Streamlit demo
├── main.py                  # Pipeline entrypoint
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Evaluation & results
- Primary metric: SMAPE (Stratified 5-fold CV)
- Example CV performance (reported):
  - SMAPE ≈ 41.5%
  - MAE ≈ 12.9 (USD)
Notes: Results depend on dataset splits, preprocessing, and feature selection.

## Key implementation notes (concise)
- The "100ml" problem: normalized units via regex in preprocessing.py to treat variations (100ml, 100 ml, 100ML) consistently.
- Target leakage prevention: OOF mean encoding for Brand — the encoding for each fold is computed using other folds only.
- Metric choice: SMAPE is preferred over RMSE to avoid over-penalizing high-priced outliers.

## Contributing
- Please open issues or pull requests. Follow repository code style and include reproducible examples.

## License
This project is licensed under the MIT License — see the LICENSE file for details.

