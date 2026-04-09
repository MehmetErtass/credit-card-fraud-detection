<div align="center">

# 💳 Credit Card Fraud Detection

### Machine Learning-Based Fraud Detection Using Random Forest

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square&logo=python&logoColor=white)](https://seaborn.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=python&logoColor=white)](https://matplotlib.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

[Overview](#-overview) ·
[Features](#-features) ·
[Tech Stack](#-tech-stack) ·
[Getting Started](#-getting-started) ·
[Usage](#-usage) ·
[Model Performance](#-model-performance) ·
[Project Structure](#-project-structure) ·
[Future Improvements](#-future-improvements) ·
[License](#-license)

</div>

---

## 📌 Overview

**Credit Card Fraud Detection** is a machine learning project designed to identify fraudulent credit card transactions using a highly imbalanced real-world dataset. The project includes **exploratory data analysis (EDA)**, **feature engineering**, **data visualization**, and **model development** using a **Random Forest Classifier**.

The dataset contains anonymized features (`V1–V28`) generated through **Principal Component Analysis (PCA)**, along with `Time`, `Amount`, and the target variable `Class`, where:

- `0` → Legitimate transaction
- `1` → Fraudulent transaction

> 🎯 **Objective:** Build a robust and reliable model capable of detecting fraudulent transactions while minimizing false positives.

---

## ✨ Features

| Module | Description |
|--------|-------------|
| **Data Exploration** | Inspection using `info()`, `describe()`, and missing value analysis |
| **Data Visualization** | Histogram and boxplots for transaction amount and class distribution |
| **Feature Engineering** | Log transformation of the `Amount` variable |
| **Modeling** | Fraud detection using `RandomForestClassifier` |
| **Evaluation** | Precision, Recall, F1-Score, Accuracy, and ROC-AUC metrics |
| **ROC Curve** | Visualization of model performance |
| **Imbalanced Data Insight** | Understanding the impact of class imbalance |

---

## 🛠 Tech Stack

| Category | Technologies |
|---------|-------------|
| **Programming Language** | Python 3.9+ |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Development Environment** | Jupyter Notebook |

---

## 🚀 Getting Started

### 📌 Prerequisites

- Python 3.9 or higher
- pip package manager
- Jupyter Notebook or any Python IDE

### 📥 Installation

```bash
# 1. Clone the repository
git clone https://github.com/mehmetertass/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 📊 Dataset

This project uses the Credit Card Fraud Detection dataset available on Kaggle:

🔗 [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

After downloading, place the `creditcard.csv` file in the root directory of the project.

---

## 📖 Usage

**1️⃣ Load the Dataset**

```python
import pandas as pd
data = pd.read_csv('creditcard.csv')
```

**2️⃣ Exploratory Data Analysis**

```python
print(data.info())
print(data.describe())
print(data.isnull().sum())
```

**3️⃣ Data Visualization**

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(data['Amount'], kde=True)
plt.title('Transaction Amount Distribution')
plt.show()

sns.boxplot(x='Class', y='Amount', data=data)
plt.title('Class vs Transaction Amount')
plt.show()
```

**4️⃣ Feature Engineering**

```python
import numpy as np
data['log_amount'] = np.log1p(data['Amount'])
```

**5️⃣ Train-Test Split**

```python
from sklearn.model_selection import train_test_split

X = data.drop(columns=['Class', 'Time'])
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```

**6️⃣ Model Training**

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

**7️⃣ Model Evaluation**

```python
from sklearn.metrics import classification_report, roc_auc_score

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Use predict_proba for a proper ROC-AUC score
print(f"ROC AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])}")
```

**8️⃣ ROC Curve Visualization**

```python
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])

plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend()
plt.show()
```

---

## 📊 Model Performance

The Random Forest Classifier demonstrates strong performance despite the highly imbalanced nature of the dataset.

| Metric | Score |
|--------|-------|
| Accuracy | ~1.00 |
| Precision (Fraud) | 0.93 |
| Recall (Fraud) | 0.80 |
| F1-Score (Fraud) | 0.86 |
| ROC-AUC | 0.90 |

> ⚠️ **Note:** Due to the severe class imbalance (~0.17% fraud), Recall and ROC-AUC are more informative metrics than overall accuracy. Results were obtained without SMOTE or resampling — applying these techniques may further improve Recall.

---

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── fraud_detection.ipynb   # Jupyter Notebook containing the full workflow
├── creditcard.csv          # Dataset (not included due to size)
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

---

## 🔮 Future Improvements

- 🔄 Handle class imbalance using SMOTE or undersampling
- ⚙️ Hyperparameter tuning with GridSearchCV or RandomizedSearchCV
- 🤖 Experiment with advanced models such as XGBoost and LightGBM
- 🌐 Deploy the model using Flask, FastAPI, or Streamlit
- 📈 Implement real-time fraud detection pipelines
- 📊 Add feature importance visualization

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## 🙋‍♂️ Author

<div align="center">

**Mehmet Ertaş**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/mehmetertas)
[![Portfolio](https://img.shields.io/badge/Portfolio-000000?style=flat-square&logo=github&logoColor=white)](https://mehmetertass.github.io)

Made with ❤️ by Mehmet Ertaş

</div>
