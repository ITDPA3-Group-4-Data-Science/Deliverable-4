# Deliverable-4

Impact of Digital Media Consumption on Academic Performance

## 📚 Overview

This project investigates how students’ digital media usage (social media hours, streaming, multitasking, using platforms for study, etc.) relates to their academic outcomes and their perception of whether digital media helps or harms them.

The repo includes:

* The original survey dataset
* A data preprocessing notebook that cleans and encodes the survey responses
* A model development & training notebook that builds machine learning models
* Saved trained models (`.pkl` files) for reuse
* `requirements.txt` to recreate the environment

This README explains how to set up the environment, run preprocessing, train and evaluate the models, and interpret the outputs.

---

## ✅ 1. Environment Setup

### 1.1 Clone / download this repo

```bash
git clone <this-repo-url>
cd Deliverable-4
```

### 1.2 Create and activate a virtual environment (recommended)

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate

# Mac / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 1.3 Install dependencies

All required libraries are pinned in `requirements.txt`.

```bash
pip install -r requirements.txt
```

This installs:

* `pandas`, `numpy` – data loading, cleaning, manipulation
* `matplotlib`, `seaborn` – plotting and visualisation
* `scikit-learn` – model training, hyperparameter tuning, metrics
* `scipy`, `joblib`, `openpyxl` (if included) – stats tests, model saving, Excel support

> Note: `openpyxl` is needed so pandas can read `.xlsx`.

---

## 📂 2. Project Files

### Data & Preprocessing

* `ITDPA3 -34 Data Preprocessing.ipynb`
* `The impact of Digital Media Consumption on...xlsx` (raw survey data)
* `dataset_encoded.xlsx` (preprocessed output)

### Model Training / Evaluation

* `ITDPA3-34 Model Development and Training - ...ipynb`
  (we have multiple versions/notebooks; they follow the same structure)

### Trained Models

* `best_model_academicAvg.pkl`
* `best_model_academicPerformance_num.pkl`
* `best_model_digitalEffect_num.pkl`

### Other

* `requirements.txt`
* `README.md`

---

## 🔄 3. Data Preprocessing Workflow

Run the preprocessing notebook first:
`ITDPA3 -34 Data Preprocessing.ipynb`

What it does:

1. **Loads the raw survey data** (student responses on media usage and academics).

2. **Selects relevant columns** such as:

   * Year of study
   * Hours on social media / streaming
   * Multitasking, binge-watching
   * Academic self-rating and average
   * Perceived impact of digital media

3. **Renames columns** to shorter, code-friendly names.

4. **Cleans the data**:

   * Removes empty or unusable rows
   * Trims whitespace
   * Handles multi-select answers (e.g. “Which platforms distract you?”)

5. **Encodes categories**:

   * Ordinal answers (e.g. “Never”, “Rarely”, “Often”, “Always”) are mapped to ordered numbers.
   * Hour ranges like `"3 - 4 hours"` are mapped to numeric midpoints like `3.5`.
   * Academic grade ranges like `"60 - 69%"` become midpoints like `65`.
   * Nominal text columns such as institution type, device, etc. are one-hot encoded (`institution_Private ... = 1/0`).
   * Multi-select fields are turned into multiple binary columns using `MultiLabelBinarizer`.

6. **Outputs the final ML-ready dataset**:

   * Saved as `dataset_encoded.xlsx`
   * All columns are numeric (`int`, `float`, or 0/1 flags)
   * No missing values
   * One row per student

This is the dataset that all models train on.

---

## 🤖 4. Model Development & Training

Run the model training notebook:
`ITDPA3-34 Model Development and Training - ...ipynb`

This notebook:

1. Loads `dataset_encoded.xlsx`
2. Picks a **target variable** to predict
3. Splits into train/test
4. Trains and tunes multiple models
5. Evaluates performance and explains the findings
6. Saves the best model for that target

We repeat this for **three different targets** to answer slightly different questions which is the three different model development files.

---

### 4.1 Targets

We train three separate regressors:

#### 1. `academicPerformance_num`

* What it is: student’s self-rated academic performance (e.g. “poor” → 1 … “excellent” → 5)
* Goal: Can we predict how students *feel* they’re performing?

#### 2. `academicAvg_num`

* What it is: student’s recent academic average in % (numeric estimate using midpoints of ranges, e.g. “60–69%” → 65).
* Goal: Can we predict actual academic results?

#### 3. `digitalEffect_num`

* What it is: student’s view of the impact of digital media on their academics
  (very negative → 1 … very positive → 5)
* Goal: Can we predict whether the student thinks digital media helps or hurts them?

We run the same pipeline three times, just changing the `TARGET` variable.

In code, it looks like:

```python
TARGET = "academicPerformance_num"
# or
TARGET = "academicAvg_num"
# or
TARGET = "digitalEffect_num"
```

---

### 4.2 Train/Test Split

We split the encoded dataset as:

* 70% for training the model
* 30% for testing (unseen data)

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)
```

This lets us report honest performance on new data.

---

### 4.3 Models Used

For each target, we train and compare:

* **Ridge Regression**
  Linear model with L2 regularisation; good for interpretability.
* **DecisionTreeRegressor**
  Non-linear splits; can capture simple threshold rules.
* **RandomForestRegressor**
  Ensemble of many decision trees; usually strongest predictive performance.

Why multiple models?
We want both accuracy (Random Forest usually wins) and interpretability (Ridge is easier to explain to a human).

---

### 4.4 Hyperparameter Tuning

Each model is tuned using `GridSearchCV` with `KFold` cross-validation (5 folds).
We search over parameter grids like:

```python
ridge_grid = {
    "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
}

dt_grid = {
    "max_depth": [3, 4, 5, 8, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
}

rf_grid = {
    "n_estimators": [200, 500],
    "max_depth": [None, 8, 12],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ["sqrt", 0.5],
}
```

Why this matters:

* We’re not just fitting one model blindly.
* We’re letting cross-validation pick the best version of each model.

---

## 📏 5. Evaluation

For each trained model, we report:

* **RMSE** (Root Mean Squared Error):
  “On average, how far off are our predictions?”
* **MAE** (Mean Absolute Error):
  “How big is the typical mistake?”
* **R²** (coefficient of determination):
  “How much of the variation in the target can we explain?”

Example printed output in the notebook:

```text
=== Random Forest ===
Best params: {...}
Test RMSE: 8.5254
Test MAE : 5.7122
Test R^2 : 0.2418
```

We also generate:

* Residual plots (how the errors behave)
* Feature importance bar charts (which inputs matter most)
* Partial dependence style plots (how a single feature affects prediction, holding others steady)
* Comparison tables across all models

These are used to interpret findings in the report.

---

## 💾 6. Saving the Best Model

After evaluating all three models (Ridge, Decision Tree, Random Forest), we keep the best-performing model for that specific target.

We save that final model to disk as a `.pkl` file using `joblib`:

```python
import joblib
joblib.dump(best_model, "best_model_academicAvg.pkl")
```

You’ll see one `.pkl` per target in the repo:

* `best_model_academicAvg.pkl`
* `best_model_academicPerformance_num.pkl`
* `best_model_digitalEffect_num.pkl`

These files let you reuse the trained model later without retraining:

```python
import joblib
model = joblib.load("best_model_academicAvg.pkl")
predictions = model.predict(new_data)
```

---

## ✅ 7. Typical Run Order (Summary)

1. Create environment and install requirements
2. Open `ITDPA3 -34 Data Preprocessing.ipynb`

   * Run all cells
   * Confirm it writes `dataset_encoded.xlsx`
3. Open `ITDPA3-34 Model Development and Training - ...ipynb`

   * Set `TARGET` to one of:

     * `"academicAvg_num"`
     * `"academicPerformance_num"`
     * `"digitalEffect_num"`
   * Run all cells
   * View printed metrics and visualisations
   * Let it save `best_model_*.pkl`
4. Repeat step 3 for each target if you want all three `.pkl` model files.

---

## 📌 8. Interpretation

* For `academicAvg_num`:
  We model actual academic average (%) as the outcome.
  Random Forest usually performs best here.
  Interpretation: higher non-academic digital media usage is linked with lower academic averages.

* For `academicPerformance_num`:
  We model self-reported performance (“Poor”, “Good”, etc.).
  Again, Random Forest tends to fit best, showing that behaviour predicts perceived performance to a moderate degree.

* For `digitalEffect_num`:
  We model whether students *think* digital media helps or harms them.
  Ridge Regression often performs best here, which is useful because we get coefficients that explain which behaviours (e.g. multitasking, distraction, binge usage) drive negative perception.

Together, these models let us say:

> Digital media usage patterns are related to both actual academic performance and how students feel digital media affects them.

---

