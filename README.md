# Steam Game Review Prediction

Machine Learning course project, University of Lahore.
Authors: Muhammad Ashhad, Urneeb Kashif (Group XX)

## Problem
Predict a Steam game's overall review class (**Positive / Mixed / Negative**) from Steam store
information available around release (price, release year, genres, categories, DLC, awards,
age rating, Mac/Linux support, developer, publisher). Awards, DLC counts and some tags can change
after launch; this is treated as a limitation. Multi-class classification on imbalanced data
(Positive 72.2%, Mixed 23.7%, Negative 4.1%).

Project questions:
1. Can Steam store information available around release predict review class better than a baseline?
2. Which of 5 models performs best (macro F1, ROC-AUC, PR-AUC)?
3. Which features matter most?

## Dataset
Steam Games sales Dataset, Kaggle (armaanmanwatkar), MIT license.
Link: https://www.kaggle.com/datasets/armaanmanwatkar/steam-games-sales-dataset
42,497 rows x 24 columns; 40,020 rows after removing games with no review label.
The file is included in `Dataset/steam-games.csv` (MIT licence); details in `Dataset/Dataset_Information.txt`.

## Method (summary)
- EDA: target and class imbalance, price, release year, correlations, outliers.
- Preprocessing: 70/30 stratified split (random_state=42) before imputation, encoding and
  scaling; train-only medians; genre/category tags used by 50+ games (re-checked on training
  rows only: same 44 tags); log1p on price/DLC/awards; multi-label tags for genres and
  categories; developer/publisher log-frequency encoding (leave-one-out on train);
  standard scaling for Logistic Regression and SVM only.
- Baselines: majority-class and stratified dummy classifiers.
- Models: Logistic Regression, Decision Tree, Random Forest, SVM (RBF), XGBoost.
  Class imbalance handled with class weights / balanced sample weights.
  Tuned with GridSearchCV, 5-fold stratified CV on the training set, scored by macro F1.
- Evaluation on the held-out test set (12,006 rows): accuracy, macro precision/recall/F1,
  per-class recall, specificity, ROC-AUC (one-vs-rest), PR-AUC, confusion matrices,
  permutation importance.

## Main results (test set)
| Model | Macro F1 | Accuracy | ROC-AUC |
|---|---|---|---|
| Random Forest | 0.470 | 0.668 | 0.724 |
| XGBoost | 0.462 | 0.654 | 0.714 |
| SVM | 0.433 | 0.583 | 0.682 |
| Decision Tree | 0.394 | 0.621 | 0.556 |
| Logistic Regression | 0.387 | 0.525 | 0.691 |
| Baseline (stratified) | 0.334 | 0.579 | - |
| Baseline (most frequent) | 0.280 | 0.722 | - |

Strongest feature: `release_year` (permutation importance, macro F1 drop 0.063).

## Folder structure
```
GroupXX_ML_Project/
├── Code/
│   ├── ML_Project.ipynb        main notebook (Sections 0-10, with saved outputs)
│   ├── requirements.txt
│   └── Python_Files/README.txt (no separate .py files were used)
├── Report/
│   └── ML_Project_Report.pdf   IEEE-format report
├── Presentation/
│   └── ML_Project_Presentation.pptx
├── Dataset/
│   ├── steam-games.csv
│   └── Dataset_Information.txt
├── README.md
└── Links.txt                   dataset, presentation (Google Drive) and live demo links
```

## How to run
1. Open `Code/ML_Project.ipynb` in Google Colab.
2. Put `Dataset/steam-games.csv` in your Drive at `Colab Notebooks/ML_Project/` (or change `DATA_PATH` in Section 0).
3. Run the notebook top to bottom **but skip Section 5b** (grid searches, over 1 hour; the best
   settings are already stored in Section 5a).
4. After a Colab disconnect do **not** use "Run all": run only Section 9 (Restore).
   **Never rerun Section 7 (Save bundle) after a disconnect.**

To run locally: `pip install -r Code/requirements.txt`, then replace the Google Drive paths.

## Environment
Google Colab, Python 3.13.18; numpy 2.4.6, pandas 2.2.3, scikit-learn 1.6.1, xgboost 3.4.1,
matplotlib 3.10.0, seaborn 0.13.2. Random seed 42 throughout.

## Live demo
https://steam-review-predictor.streamlit.app/
