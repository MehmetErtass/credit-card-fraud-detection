# Credit Card Fraud Detection

In this project, I use **machine learning** to detect **credit card fraud** using a dataset that contains anonymized credit card transaction data. The goal is to identify fraudulent transactions based on various features. I built a predictive model using a **Random Forest Classifier** and evaluated its performance with metrics like **ROC AUC**, **precision**, and **recall**.

## Project Overview

This project aims to predict whether a credit card transaction is fraudulent. The dataset is sourced from **Kaggle’s Credit Card Fraud Detection** challenge, which consists of over 280,000 transactions.

### Key Steps:

1. **Data Preprocessing**: 
    - Explored the dataset to understand basic features, check for missing values, and generate statistical summaries.
    - Handled any missing data and performed **feature engineering**. For instance, I used log transformation on the transaction amount to handle outliers more effectively.

2. **Model Building**:
    - Split the dataset into training and testing sets.
    - Trained a **Random Forest Classifier** on the training data.
    - Evaluated the model’s performance using metrics like **ROC AUC**, **accuracy**, **precision**, and **recall**.

3. **Model Evaluation**:
    - Visualized the model’s performance with an **ROC curve**.
    - Analyzed the false positives and true positives to assess how robust the model is.

## Dataset

The dataset used is **Credit Card Fraud Detection**, available on Kaggle, and it includes the following features:
- `Time`: Time since the first transaction.
- `V1-V28`: Anonymized features derived from PCA (Principal Component Analysis).
- `Amount`: The transaction amount.
- `Class`: Label (1 for fraud, 0 for non-fraud).

## Tools and Libraries Used

- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computing.
- **Matplotlib & Seaborn**: For creating visualizations.
- **Scikit-Learn**: For building the machine learning model and evaluating performance.

## How to Run the Project

If you'd like to run this on your local machine, here are the steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/MehmetErtass/credit-card-fraud-detection.git
    ```

2. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset (`creditcard.csv`) and place it in the project folder.

4. Run the Python script to train the model and analyze the data:
    ```bash
    python fraud_detection.py
    ```

## Results

- **Model Performance**: The Random Forest Classifier achieved an **ROC AUC score** of X.XX, showing its ability to differentiate between fraudulent and non-fraudulent transactions effectively.
- **Visualizations**: 
    - Distribution of transaction amounts.
    - Boxplot relating the transaction amount to whether the transaction was fraudulent or not.
    - ROC curve for evaluating the model’s performance.

## Future Improvements

1. **Hyperparameter Tuning**: I plan to fine-tune the model using tools like **GridSearchCV** or **RandomizedSearchCV** to improve its performance.
2. **Feature Engineering**: I’ll explore additional features and apply more advanced feature selection techniques to enhance accuracy.
3. **Trying Other Models**: I’m also considering experimenting with other classification algorithms, such as **XGBoost**, **LightGBM**, and **Logistic Regression**, to see if they can provide better results.

## Acknowledgements

- Dataset: Kaggle’s **Credit Card Fraud Detection** dataset.
- Libraries: Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn.
