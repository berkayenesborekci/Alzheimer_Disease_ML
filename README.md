# Alzheimer's Disease Classification

A supervised machine learning project that compares Logistic Regression, Random Forest, and XGBoost for predicting Alzheimer's disease diagnosis from patient clinical data.

---

## 👥 Team Members

| Name | Role | Contribution |
|------|------|-------------|
| Berkay Enes Börekci | Project Coordinator | Data pipeline (`data_pipeline.ipynb`), documentation (`requirements.txt`, `README.md`) |
| Hayrettin Calban | Model Developer | Model definitions (`model.ipynb`), training loop (`train.ipynb`) |
| Resul Kaplan | Pipeline Engineer | End-to-end pipeline notebook (`main.ipynb`) |
| Emir İbrahim Polat | Evaluation Specialist | Model evaluation, metrics & visualizations (`evaluate.ipynb`) |

---

## 📊 Dataset

- **Source:** [Alzheimer's Disease Dataset — Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset)
- **Size:** 2,149 patients × 35 columns
- **Target variable:** `Diagnosis` (0 = Healthy, 1 = Alzheimer's)
- **Class distribution:** ~64.7% Healthy / ~35.3% Alzheimer's

### Feature Groups

| Group | Features |
|-------|----------|
| Demographics | Age, Gender, Ethnicity, EducationLevel |
| Lifestyle | BMI, Smoking, AlcoholConsumption, PhysicalActivity, DietQuality, SleepQuality |
| Medical History | FamilyHistoryAlzheimers, CardiovascularDisease, Diabetes, Depression, HeadInjury, Hypertension |
| Clinical Measurements | SystolicBP, DiastolicBP, CholesterolTotal, CholesterolLDL, CholesterolHDL, CholesterolTriglycerides |
| Cognitive & Functional | MMSE, FunctionalAssessment, MemoryComplaints, BehavioralProblems, ADL |
| Symptoms | Confusion, Disorientation, PersonalityChanges, DifficultyCompletingTasks, Forgetfulness |

---

## 🔧 Preprocessing

1. **Dropped irrelevant columns:** `PatientID` (sequential row number, no predictive value) and `DoctorInCharge` (constant value across all patients)
2. **Target separation:** `Diagnosis` column extracted as `y`; remaining columns used as features `X`
3. **Categorical encoding:** `pd.get_dummies(drop_first=True)` converts text/category columns to binary (0/1) and removes one redundant dummy per category to avoid multicollinearity
4. **Train/Test Split:** 80% training / 20% testing with `stratify=y` to preserve the class ratio in both splits (`random_state=42`)
5. **Feature Scaling:** `StandardScaler` fitted **only on training data** and applied to both splits — fitting on test data would be data leakage

---

## 🤖 Models

| Model | Key Parameters | Notes |
|-------|---------------|-------|
| Logistic Regression | `max_iter=1000` | Linear baseline; trained on **scaled** data |
| Random Forest | `n_estimators=200, n_jobs=-1` | 200 trees, majority vote; trained on **raw** data |
| XGBoost | `n_estimators=300, max_depth=6, learning_rate=0.1` | Boosting; each tree corrects previous errors; **raw** data |

> Tree-based models (Random Forest, XGBoost) are scale-invariant; Logistic Regression requires standardized features.

---

## 📈 Results

| Model | Accuracy | vs. Baseline |
|-------|----------|-------------|
| Majority-class Baseline | 64.65% | — |
| Logistic Regression | 81.63% | +16.98% |
| Random Forest | **94.42%** | +29.77% |
| XGBoost | **94.42%** | +29.77% |

- All three models significantly outperform the majority-class baseline.
- Random Forest and XGBoost reach identical accuracy; they differ on 8 individual predictions that cancel out.
- Evaluation includes **accuracy**, **precision**, **recall**, **F1-score**, and **confusion matrices** for each model.
- **Top predictive features** identified via Random Forest feature importances (see `figures/feature_importance.png`).

---

## 📁 Project Structure

```
ALZHEIMER_ML_PROJECT/
│
├── data/
│   └── alzheimers_disease_data.csv   # Raw dataset
│
├── figures/
│   ├── confusion_matrices.png        # Side-by-side confusion matrices
│   ├── feature_importance.png        # Top 15 features (Random Forest)
│   └── accuracy_comparison.png       # Bar chart comparing model accuracies
│
├── src/
│   ├── data_pipeline.ipynb           # Data loading & preprocessing (step-by-step)
│   ├── model.ipynb                   # Model definitions explained individually
│   ├── train.ipynb                   # Training loop & predictions
│   └── evaluate.ipynb                # Full evaluation with all metrics & plots
│
├── main.ipynb                        # End-to-end pipeline (single notebook)
├── PROJECT_EXPLAINED.txt             # Line-by-line explanation of every cell
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## ⚙️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
Open `main.ipynb` in Jupyter and run all cells top-to-bottom (Shift+Enter).

### 3. Run step-by-step notebooks (optional)
Run in order from inside the `src/` folder:
```
data_pipeline.ipynb → model.ipynb → train.ipynb → evaluate.ipynb
```

### Requirements
```
pandas==2.2.3
numpy==2.2.3
scikit-learn==1.6.1
xgboost==3.2.0
matplotlib==3.10.1
seaborn==0.13.2
```
