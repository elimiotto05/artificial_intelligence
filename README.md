# Tennis Match Outcome Prediction

The goal of the project is to predict the winner of a professional tennis match using information available **before the match is played**.

The dataset contains ATP matches from **2015 to 2025**. Starting from the original match data, we created a machine-learning pipeline that includes data cleaning, feature engineering, temporal data splitting, model training, model comparison and final evaluation.

The target variable is `Player1Won`, which indicates whether Player 1 won the match.

## Project Overview

The original datasets represent matches using a winner/loser structure. Since using this representation directly could introduce data leakage, the matches were converted into a neutral `Player1` / `Player2` representation.

We then created pre-match features based on information such as:

* ATP ranking and ranking points
* previous matches played
* overall historical win rate
* recent form
* surface-specific performance
* head-to-head history
* days since the previous match

Bookmaker odds are available in the original dataset, but they are **not used as features by the machine-learning models**. They are used only at the end of the project as an external benchmark.

## Models

Six different classification models were developed and compared:

* Artificial Neural Network
* Random Forest
* k-Nearest Neighbors
* AdaBoost
* XGBoost
* LightGBM

Model development was performed using a chronological approach in order to respect the temporal nature of the data.

The main development split is:

```text
2015–2022  → model development
2023       → external validation
2024–2025  → final untouched test set
```

Temporal cross-validation was also used during model development, always training on past matches and validating on future matches.

## Final Results

Among the tested models, **XGBoost obtained the best overall performance** on the final 2024–2025 test set.

| Model         | Accuracy | ROC-AUC |
| ------------- | -------: | ------: |
| XGBoost       |   65.09% |   0.712 |
| AdaBoost      |   64.80% |   0.710 |
| LightGBM      |   64.76% |   0.709 |
| Random Forest |   64.66% |   0.710 |
| ANN           |   64.35% |   0.703 |
| k-NN          |   64.23% |   0.699 |

For comparison, predicting the **better-ranked player** gives an accuracy of approximately **64.25%** on the final test set.

On the subset of matches for which bookmaker odds are available, the bookmaker benchmark reaches approximately **68.50% accuracy**.

A more detailed discussion of the methodology, experiments and results is provided in the project report.

## Repository Structure

```text
artificial_intelligence/
│
├── data/
│   ├── raw/                  # Original yearly datasets (2015–2025)
│   └── processed/            # Cleaned and engineered datasets
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_data_split.ipynb
│   ├── 05_ann_classifier.ipynb
│   ├── 06_random_forest_classifier.ipynb
│   ├── 07_knn_classifier.ipynb
│   ├── 08_adaboost_classifier.ipynb
│   ├── 09_xgboost_classifier.ipynb
│   ├── 10_lightgbm_classifier.ipynb
│   ├── 11_feature_importance_analysis.ipynb
│   ├── 12_final_model_training.ipynb
│   └── 13_final_evaluation.ipynb
│
├── src/
│   ├── preprocessing.py      # Shared preprocessing functions
│   ├── evaluation.py         # Model evaluation metrics
│   └── temporal_cv.py        # Temporal cross-validation utilities
│
├── models/                   # Saved validation and final models
│
├── results/
│   ├── model_selection/      # Cross-validation and tuning results
│   ├── validation/           # Validation results
│   ├── feature_importance/   # Feature-importance analyses
│   └── final/                # Final test results and predictions
│
├── requirements.txt
└── README.md
```

## Notebook Workflow

The notebooks are intended to be followed in numerical order.

**01–04:** data understanding, cleaning, feature engineering and temporal splitting.

**05–10:** development and comparison of the six machine-learning models.

**11:** feature-importance analysis.

**12:** training of the final versions of the models using the selected configurations.

**13:** final evaluation on the untouched 2024–2025 test period.

## Installation

To install the required Python packages:

```bash
pip install -r requirements.txt
```

The main dependencies include:

```text
numpy
pandas
matplotlib
scikit-learn
tensorflow
xgboost
lightgbm
openpyxl
joblib
```

## Reproducing the Project

To reproduce the complete workflow, run the notebooks in numerical order starting from:

```text
01_data_understanding.ipynb
```

and ending with:

```text
13_final_evaluation.ipynb
```

Intermediate processed datasets, trained models and result files are saved in the corresponding `data/processed`, `models` and `results` folders.

## Authors

Developed by Harbas Amina, Miotto Elisa, Ponga Agnese.
