# Customer Churn Prediction - Telco Dataset

This project focuses on predicting customer churn for a telecom company using machine learning models. Churn refers to customers who stop using the service. Identifying these customers in advance can help the business take preventive actions.

---

## Problem Statement

Telecom companies often struggle with customer retention. The goal of this project is to predict whether a customer is likely to churn based on various demographic and service-related features.

---

## Dataset

- Source: Kaggle - Telco Customer Churn
- Target variable: `Churn` (Yes/No)
- Features include:
  - Gender
  - SeniorCitizen
  - Tenure
  - MonthlyCharges
  - TotalCharges
  - Contract Type
  - Payment Method
  - Internet Service, etc.

---

## Objectives

- Understand and clean the dataset
- Explore patterns in customer churn
- Encode and scale relevant features
- Build and evaluate classification models
- Identify important features contributing to churn

---

## Tools and Libraries Used

- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## Process Overview

1. **Exploratory Data Analysis (EDA)**  
   Basic statistics and visualizations to understand the data.

2. **Data Preprocessing**  
   Converted TotalCharges to numeric, handled missing values, encoded categorical variables, and scaled numerical features.

3. **Model Training**  
   Built and evaluated classification models like Logistic Regression and Random Forest.

4. **Model Evaluation**  
   Used accuracy, precision, recall, F1-score, and confusion matrix to measure performance.

---

## Results

- The Random Forest model performed the best with accuracy around 80%.
- Features like Contract type, Tenure, and MonthlyCharges were strong indicators of churn.

---

## How to Run

1. Clone this repository or download the `.ipynb` file.
2. Install the necessary libraries if not already installed:
3. Run the notebook in Jupyter or Google Colab.

---

## Future Work

- Experiment with other models like XGBoost or LightGBM.
- Perform hyperparameter tuning (e.g., GridSearchCV).
- Build a dashboard using Streamlit or Flask.

---

## About

This project was created as part of a personal learning journey in data science and machine learning. It aims to demonstrate the application of core concepts such as classification, feature engineering, and model evaluation on a real-world business problem.
