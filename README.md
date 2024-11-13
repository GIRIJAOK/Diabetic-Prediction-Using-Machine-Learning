# Diabetic Prediction Using Machine Learning

This project focuses on predicting whether a person has diabetes based on various medical features. It utilizes machine learning algorithms to analyze data and provide predictions. The model is trained on a dataset of patients' health data, including factors like age, BMI, glucose levels, and more.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [How to Use](#how-to-use)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

Diabetes is a chronic disease that occurs when the body is unable to properly regulate blood sugar levels. The goal of this project is to build a predictive model that can help in early detection of diabetes based on input health metrics. This can assist healthcare professionals in diagnosing patients and planning early interventions.

The project leverages machine learning algorithms, including logistic regression, decision trees, and support vector machines (SVM), to predict whether a person has diabetes or not. The dataset is pre-processed, and various models are evaluated based on accuracy, precision, recall, and F1-score.

## Dataset

This project uses the **Pima Indians Diabetes Dataset**, which contains data from 768 women of Pima Indian heritage, collected in the 1990s. The dataset includes the following features:

- **Pregnancies**: Number of pregnancies
- **Glucose**: Plasma glucose concentration after 2 hours in an oral glucose tolerance test
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg / height in m²)
- **DiabetesPedigreeFunction**: A function that represents the genetic relationship between diabetes and the patient
- **Age**: Age (years)
- **Outcome**: Whether or not the patient has diabetes (0 = No, 1 = Yes)

The dataset can be found at the following link:
[Download Dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)

### Example Usage

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("diabetes.csv")

# Data Preprocessing
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and Evaluate the Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## Model Evaluation

After training the model, evaluation is performed using accuracy, precision, recall, F1-score, and confusion matrix to assess the performance of the prediction. The `classification_report` from `sklearn.metrics` provides a summary of these metrics.

### Example Output:

```bash
Accuracy: 0.7727272727272727
              precision    recall  f1-score   support

           0       0.76      0.87      0.81       97
           1       0.79      0.64      0.71       50

    accuracy                           0.77      147
   macro avg       0.77      0.75      0.76      147
weighted avg       0.77      0.77      0.77      147
```

## Results

The results show that the logistic regression model performs reasonably well in predicting diabetes outcomes. The model achieves an accuracy of around 77%, with a good balance of precision and recall. Additional models like **Support Vector Machines (SVM)** and **Decision Trees** can also be experimented with to compare performance.

### Additional Features to Explore:

- **Hyperparameter Tuning**: Optimize model parameters using grid search or random search for better results.
- **Cross-Validation**: Implement cross-validation techniques to validate the model performance more robustly.
- **Feature Engineering**: Experiment with adding/removing features to improve prediction accuracy.
