# -*- coding: utf-8 -*-
"""
# Diabetes Prediction (Classification) — Pima Indians Diabetes (Kaggle)

This end-to-end project predicts **onset of diabetes** from diagnostic measurements using the **Pima Indians Diabetes** dataset, which is available on Kaggle (mirrored from the original UCI dataset).

**What you'll find inside:**
- Data loading (local / public mirror / Kaggle API optional)
- Quick EDA (shapes, class balance, distributions)
- Data cleaning: treat biologically-impossible zeros as missing and impute
- Feature scaling (for certain models)
- Baseline & advanced models (Logistic Regression, KNN, SVM, Random Forest)
- Cross-validation & model selection
- Evaluation: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix
- Hyperparameter tuning with `GridSearchCV`
- Save the best model (`.pkl`) and show how to use it for inference

> ⚠️ If you prefer to load directly from Kaggle, see the **Kaggle API (optional)** cell below.
"""

# Python version & package setup (uncomment if needed)
import sys, sklearn, numpy as np, pandas as pd, matplotlib
print("Python:", sys.version)
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("matplotlib:", matplotlib.__version__)
import sklearn
print("scikit-learn:", sklearn.__version__)

# Plotting inline (in a script, plots will show in a separate window)
# %matplotlib inline

"""
## Load the Dataset

We first try to load a local `diabetes.csv`. If it's not found, we fall back to a reliable public mirror of the same dataset (originally UCI, also on Kaggle as **Pima Indians Diabetes Database**).  
Finally, there's an **optional** Kaggle API cell below if you want to pull directly from Kaggle.
"""

import os
import pandas as pd

CSV_CANDIDATES = [
    # 1) Local file if you already have it (e.g., downloaded from Kaggle)
    "diabetes.csv",
    # 2) Public mirror (equivalent to Kaggle/ UCI dataset; no headers in source; we'll assign names)
    #    Source: https://github.com/jbrownlee/Datasets
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
]

COLUMN_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

def load_data():
    for path in CSV_CANDIDATES:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                # If it already has headers, harmonize columns if necessary
                if set(COLUMN_NAMES).issubset(df.columns):
                    df = df[COLUMN_NAMES]
                else:
                    # Try to rename common Kaggle header variants
                    rename_map = {
                        "DiabetesPedigreeFunction": "DiabetesPedigreeFunction",
                        "SkinThickness": "SkinThickness",
                        "BloodPressure": "BloodPressure"
                    }
                    for k, v in rename_map.items():
                        if k not in df.columns and v in df.columns:
                            pass
                    # If columns do not match, try fallback
                    df.columns = COLUMN_NAMES
                print("Loaded local diabetes.csv")
                return df
            else:
                # try reading as URL (Brownlee mirror, no header)
                if path.startswith("http"):
                    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
                    print("Loaded dataset from public mirror:", path)
                    return df
        except Exception as e:
            print(f"Failed to load from {path}: {e}")
    raise RuntimeError("Failed to load dataset from all sources. Please add 'diabetes.csv' to this folder.")
    
df = load_data()
print(df.head())


"""
### (Optional) Load via Kaggle API

If you want to fetch the dataset directly from Kaggle, set up the Kaggle API:
1. Install and authenticate Kaggle (`pip install kaggle`), place your `kaggle.json` in `~/.kaggle/`.
2. Then run the code below (uncomment) to download and unzip the dataset.
"""

# OPTIONAL: Kaggle API download (uncomment to use)
# import os, zipfile, subprocess
# os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
# # Make sure kaggle.json is present at ~/.kaggle/kaggle.json before running
# # Example dataset slug: "uciml/pima-indians-diabetes-database"
# subprocess.run(["kaggle", "datasets", "download", "-d", "uciml/pima-indians-diabetes-database"], check=True)
# with zipfile.ZipFile("pima-indians-diabetes-database.zip", "r") as zf:
#     zf.extractall(".")
# # Kaggle file is typically named "diabetes.csv"
# import pandas as pd
# df = pd.read_csv("diabetes.csv")
# print(df.head())


"""
## Exploratory Data Analysis (EDA)
"""

import pandas as pd
import numpy as np

print("Shape:", df.shape)
print("\nColumns:", list(df.columns))

print("\nClass balance (Outcome):")
print(df['Outcome'].value_counts(normalize=True))

print("\nSummary stats:")
print(df.describe())


import matplotlib.pyplot as plt

# Target distribution
df['Outcome'].value_counts().plot(kind='bar')
plt.title("Outcome counts (0=No Diabetes, 1=Diabetes)")
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.show()

# Feature distributions
for col in df.columns[:-1]:
    df[col].plot(kind='hist', bins=30, alpha=0.7)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()


"""
## Data Cleaning & Preprocessing

For certain features in this dataset, **zeros** are not physiologically meaningful and actually indicate **missing values** (e.g., `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`).  
We'll convert zeros to `NaN` for those columns and then **impute** missing values with the median (within a pipeline).
"""

from sklearn.model_selection import train_test_split

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Columns where zeros indicate missingness
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


"""
## Modeling

We will train a few common classifiers:

- **Logistic Regression** (baseline, interpretable)
- **K-Nearest Neighbors**
- **Support Vector Machine**
- **Random Forest**

We'll use Scikit-Learn **pipelines** to ensure consistent preprocessing:
1. Replace zeros with NaN in selected columns
2. Impute missing values with median
3. Standardize features (when beneficial)
"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class ZeroToNaN(BaseEstimator, TransformerMixin):
    """Custom transformer: replaces zeros with NaN for specified columns."""
    def __init__(self, cols):
        self.cols = cols
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        Xc = X.copy()
        for c in self.cols:
            Xc[c] = Xc[c].replace(0, np.nan)
        return Xc


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

numeric_features = list(X.columns)

preprocess = Pipeline(steps=[
    ('zero_to_nan', ZeroToNaN(zero_as_missing)),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocess_only = ColumnTransformer(
    transformers=[('num', preprocess, numeric_features)]
)


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

models = {
    "LogisticRegression": LogisticRegression(max_iter=200, random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42)
}

pipelines = {name: Pipeline(steps=[('prep', preprocess_only), ('clf', model)])
             for name, model in models.items()}
print(pipelines)


from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, auc,
                             precision_recall_curve)

def evaluate_model(pipe, X_tr, y_tr, X_te, y_te, name="Model"):
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    y_prob = pipe.predict_proba(X_te)[:, 1] if hasattr(pipe, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall": recall_score(y_te, y_pred, zero_division=0),
        "f1": f1_score(y_te, y_pred, zero_division=0)
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_te, y_prob)
        except Exception:
            metrics["roc_auc"] = np.nan

    print(f"\n{name} performance:")
    for k, v in metrics.items():
        print(f"{k:>10}: {v:0.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_te, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot(values_format='d')
    plt.title(f"Confusion Matrix — {name}")
    plt.show()

    # ROC Curve
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:0.3f})")
        plt.plot([0,1], [0,1], linestyle='--')
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()

    # Precision-Recall Curve
    if y_prob is not None:
        precision, recall, _ = precision_recall_curve(y_te, y_prob)
        plt.plot(recall, precision, label=name)
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.show()

    return metrics


results = {}
for name, pipe in pipelines.items():
    metrics = evaluate_model(pipe, X_train, y_train, X_test, y_test, name=name)
    results[name] = metrics

print(pd.DataFrame(results).T)


"""
## Cross-Validation (Stratified K-Fold)
"""

from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {}
for name, pipe in pipelines.items():
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')
    cv_scores[name] = {"roc_auc_mean": scores.mean(), "roc_auc_std": scores.std()}
    
print(pd.DataFrame(cv_scores).T)


"""
## Hyperparameter Tuning (GridSearchCV)
We'll tune **SVM** and **RandomForest** and pick the better one by ROC-AUC.
"""

from sklearn.model_selection import GridSearchCV

param_grid_svm = {
    'clf__C': [0.1, 1, 5, 10],
    'clf__kernel': ['rbf', 'linear'],
    'clf__gamma': ['scale', 'auto']
}
pipe_svm = pipelines["SVM"]
gsvm = GridSearchCV(pipe_svm, param_grid=param_grid_svm, cv=5, scoring='roc_auc', n_jobs=-1)
gsvm.fit(X_train, y_train)

print("Best SVM params:", gsvm.best_params_)
print("Best SVM ROC-AUC (CV):", gsvm.best_score_)
svm_test_auc = roc_auc_score(y_test, gsvm.predict_proba(X_test)[:,1])
print("SVM ROC-AUC (Test):", svm_test_auc)


param_grid_rf = {
    'clf__n_estimators': [200, 300, 500],
    'clf__max_depth': [None, 4, 6, 8, 12],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4]
}
pipe_rf = pipelines["RandomForest"]
grf = GridSearchCV(pipe_rf, param_grid=param_grid_rf, cv=5, scoring='roc_auc', n_jobs=-1)
grf.fit(X_train, y_train)

print("Best RF params:", grf.best_params_)
print("Best RF ROC-AUC (CV):", grf.best_score_)
rf_test_auc = roc_auc_score(y_test, grf.predict_proba(X_test)[:,1])
print("RF ROC-AUC (Test):", rf_test_auc)


# Pick the better model on test ROC-AUC
best_model = gsvm if svm_test_auc >= rf_test_auc else grf
best_name = "SVM" if svm_test_auc >= rf_test_auc else "RandomForest"
print("Selected best model:", best_name)

# Evaluate thoroughly
_ = evaluate_model(best_model.best_estimator_, X_train, y_train, X_test, y_test, name=f"Best ({best_name})")

# Save the trained model
import joblib
joblib.dump(best_model.best_estimator_, "diabetes_best_model.pkl")
print("Saved model to diabetes_best_model.pkl")


"""
## Feature Importance / Coefficients

For tree-based models we can inspect feature importances; for linear models we can view coefficients.  
(For SVM with RBF kernel, feature importances are not directly available.)
"""

from sklearn.inspection import permutation_importance

final_pipe = best_model.best_estimator_
clf = final_pipe.named_steps['clf']

try:
    # If it's a tree-based model with feature_importances_
    importances = getattr(clf, "feature_importances_", None)
    if importances is not None:
        # Get feature names after preprocessing (same as original numeric_features)
        feat_names = list(X.columns)
        fi = pd.Series(importances, index=feat_names).sort_values(ascending=False)
        print(fi)
        fi.plot(kind='bar')
        plt.title("Feature Importances")
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.show()
    else:
        raise AttributeError
except Exception:
    # Use permutation importance as a model-agnostic alternative
    r = permutation_importance(final_pipe, X_test, y_test, n_repeats=10, random_state=42)
    feat_names = list(X.columns)
    pi = pd.Series(r.importances_mean, index=feat_names).sort_values(ascending=False)
    print(pi)
    pi.plot(kind='bar')
    plt.title("Permutation Importances (Test Set)")
    plt.xlabel("Feature")
    plt.ylabel("Mean Importance")
    plt.show()


"""
## Inference: Using the Saved Model

Below is an example of how to load the saved model and run a prediction on a single patient.
"""

import numpy as np, pandas as pd, joblib

pipe = joblib.load("diabetes_best_model.pkl")

# Example patient (values are placeholders; replace with real measurements)
example = pd.DataFrame([{
    "Pregnancies": 2,
    "Glucose": 130,
    "BloodPressure": 70,
    "SkinThickness": 25,
    "Insulin": 100,
    "BMI": 28.5,
    "DiabetesPedigreeFunction": 0.45,
    "Age": 35
}])

proba = pipe.predict_proba(example)[:, 1][0]
pred = pipe.predict(example)[0]
print(f"Predicted probability of diabetes: {proba:0.3f}")
print("Predicted class:", int(pred))


"""
## Next Steps / Ideas
- Try **calibration** (e.g., `CalibratedClassifierCV`) to improve probability estimates.
- Add **threshold tuning** using PR curves or cost-sensitive metrics.
- Evaluate additional models (e.g., XGBoost, LightGBM) if available in your environment.
- Explore **class imbalance** techniques (e.g., SMOTE) if needed.
- Package this notebook into a simple **Streamlit**/**Gradio** app for interactive predictions.
"""