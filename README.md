# 📊 Bank Marketing Subscription Prediction

## 📌 Project Overview

This project predicts whether a customer will subscribe to a bank term deposit based on historical marketing campaign data.

Using demographic, financial, and campaign-related features, multiple machine learning classification models were trained and evaluated to identify customers with a high probability of subscription.

The goal is to help banks optimize marketing campaigns by targeting the right customers and improving conversion rates.

---

## 🎯 Problem Statement

Banks conduct direct marketing campaigns via phone calls.  
Each call has a cost.

Can we predict in advance which customers are more likely to subscribe to a term deposit?

This project treats the problem as a **binary classification task**:
- 1 → Customer subscribes
- 0 → Customer does not subscribe

---

## 📊 Dataset Information

- Source: Bank Marketing Dataset
- Total records: 45,000+
- Features include:
  - Demographic data (age, job, marital status, education)
  - Financial data (balance, loans)
  - Campaign details (number of contacts, previous outcomes)
- Target variable: `y` (yes/no subscription)

---

## 🔍 Data Preprocessing

The following steps were performed:

- Converted target variable to binary (yes=1, no=0)
- Removed data leakage feature (`duration`)
- One-hot encoded categorical features
- Scaled numerical features using StandardScaler
- Performed stratified train-test split

Proper preprocessing ensures realistic and unbiased model performance.

---

## 🤖 Models Implemented

The following classification models were trained and compared:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest
- Naive Bayes

---

## 📈 Evaluation Metrics

Models were evaluated using:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1 Score
- ROC-AUC Score

Since the dataset is imbalanced, ROC-AUC and Recall were carefully analyzed instead of relying only on accuracy.

---

## 🛠️ Tools & Technologies

- Python
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn
- Google Colab / Jupyter Notebook

---

## 📂 Project Structure

```
bank-marketing-subscription-prediction/
│
├── data/
├── notebooks/
├── models/
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run

1. Clone the repository:

```
git clone https://github.com/yourusername/bank-marketing-subscription-prediction.git
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the notebook or Python script.

---

## 🔮 Future Improvements

- Hyperparameter tuning (GridSearchCV)
- Handling class imbalance using class weights or SMOTE
- Gradient Boosting models (XGBoost, LightGBM)
- Feature importance visualization
- Model deployment using Flask or FastAPI

---

## 🏁 Conclusion

This project demonstrates an end-to-end machine learning pipeline:
Data Exploration → Feature Engineering → Model Training → Evaluation

The final model helps identify high-probability customers, enabling more efficient and cost-effective marketing campaigns.

## Author
SHYAM ROSH NK
