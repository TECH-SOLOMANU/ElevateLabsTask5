# 🧭 Titanic Dataset — Exploratory Data Analysis (EDA)

## 🎯 Objective
Perform **Exploratory Data Analysis (EDA)** on the **Titanic dataset** to identify patterns, trends, and relationships among passenger features that influenced survival.  
This README documents the process, findings, and deliverables for the EDA project.

---

## 🧰 Tools & Technologies
- **Python 3.x**
- **Jupyter Notebook**
- **Pandas** – Data manipulation and exploration  
- **NumPy** – Numerical operations  
- **Matplotlib** – Basic plotting and visualization  
- **Seaborn** – Advanced statistical plotting  

---

## 📂 Dataset
Dataset: **Titanic - Machine Learning from Disaster**  
Files used (examples):
- `train.csv`  — primary dataset for EDA
- `test.csv`   — (optional) for reference

Source: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data) or `sns.load_dataset('titanic')` for a built-in alternative.

---

## 🧩 Project Steps

### 1) Data Loading & Initial Inspection
- Load datasets:
```python
import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
```
- Quick checks:
```python
train.info()
train.head()
train.describe(include='all')
train.isnull().sum()
```
**Purpose:** Understand structure, data types, and missing values.

---

### 2) Data Cleaning & Missing Values
- Identify missing values:
```python
missing = train.isnull().sum().sort_values(ascending=False)
missing[missing > 0]
```
- Strategy:
  - `Cabin`: too many missing → drop or treat as a separate category.
  - `Age`: impute (median or group-wise median by `Pclass` and `Sex`).
  - `Embarked`: fill with mode (usually 'S').
  - `Fare`: check for zeros/outliers; consider log-transform if skewed.

---

### 3) Statistical Summaries & Value Counts
Use `.describe()`, `.info()`, and `.value_counts()`:
```python
train['Sex'].value_counts()
train['Pclass'].value_counts()
train['Embarked'].value_counts(dropna=False)
```
**Notes:** These quick counts reveal class imbalance and category distributions.

---

### 4) Visual Explorations
Recommended visualizations (all included in the notebook):
- **Pairplot** — `sns.pairplot(...)` (sampled if dataset large)
- **Heatmap (correlation)** — `sns.heatmap(train.corr(), annot=True)`
- **Histograms** — e.g., `sns.histplot(train['Age'], kde=True)`
- **Boxplots** — e.g., `sns.boxplot(x='Pclass', y='Fare', data=train)`
- **Countplots** — e.g., `sns.countplot(x='Sex', hue='Survived', data=train)`
- **Scatterplots** — e.g., `sns.scatterplot(x='Age', y='Fare', hue='Survived', data=train)`

Each plot should be followed by a short observation summarizing the visual insight.

---

### 5) Key Observations (Summary)
- **Sex:** Females had a significantly higher survival rate than males.
- **Pclass:** Passengers in 1st class had higher survival odds; 3rd class had the lowest.
- **Fare:** Higher fares generally correlate with higher survival probability.
- **Age:** Children and younger passengers showed slightly higher survival rates; `Age` interacts with `Sex` and `Pclass`.
- **Embarked:** Slight variations by port; passengers from Cherbourg (C) tended to have marginally better survival.
- **Cabin:** Sparse and noisy — not reliable without feature engineering.

---

### 6) Feature Engineering (Suggestions)
- Extract `Title` from `Name` (Mr, Mrs, Miss, Master, etc.) — helpful for survival patterns.
- Create `FamilySize = SibSp + Parch + 1`.
- Bin `Age` into categories (child, teen, adult, senior).
- Encode missing `Cabin` as 'Unknown' and extract deck letter when available.

---

### 7) Deliverables
- `Titanic_EDA.ipynb` — Jupyter Notebook with code, visuals, and markdown observations.
- `Titanic_EDA_Report.pdf` — PDF summary of findings and visual highlights (exported from notebook).
- `README.md` — This file documenting the project.

---

## 🚀 How to Run the Notebook
1. Ensure Python 3.x and the required packages are installed:
```bash
pip install pandas matplotlib seaborn notebook
```
2. Launch Jupyter Notebook:
```bash
jupyter notebook
```
3. Open `Titanic_EDA.ipynb` and run cells sequentially.

---

## ❓ Sample Interview Questions
1. Which features were most predictive of survival and why?  
2. How did you handle missing data in the dataset? Which imputation methods did you use and why?  
3. What new features would you create to improve model performance and why?

---

## 🧠 Limitations & Next Steps
- **Limitations:** Missing values (Cabin), limited features, and potential sampling bias.
- **Next Steps:** Feature engineering, model building (classification), cross-validation, and model explainability (SHAP/LIME).

---

## 👨‍💻 Author
**Soloman Meriga**  
📧 solomanmeriga111@gmail.com  
GitHub: https://github.com/TECH-SOLOMANU
