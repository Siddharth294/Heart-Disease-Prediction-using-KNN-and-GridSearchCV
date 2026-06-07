# ❤️ Heart Disease Prediction using KNN and GridSearchCV

## 📌 Overview

This project focuses on building a machine learning model to predict the presence of heart disease based on patient medical data. The model uses the **K-Nearest Neighbors (KNN)** algorithm and improves performance through **hyperparameter tuning using GridSearchCV**.

The objective is to support early detection of heart disease by analyzing key health indicators and classifying whether a patient is at risk.

---

## 🛠️ Tech Used

* Python 🐍
* Jupyter Notebook
* Scikit-learn
* NumPy
* Pandas
* Seaborn

---

## 📚 Libraries Used

```python
import pandas as pd
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
```

---

## 📊 Dataset

Dataset used:
- 1-heart.csv

Target Column:
- target
  - 1 = Heart Disease Present
  - 0 = No Heart Disease

The dataset contains medical information such as:
- Age
- Cholesterol
- Blood Pressure
- Heart Rate
- Chest Pain Type
- And other health-related features

### Target Variable:

* `0` → No heart disease
* `1` → Heart disease present

---
## Project Workflow

1. Import the dataset
2. Split features and target variable
3. Perform train-test split
4. Apply feature scaling using StandardScaler
5. Train KNN model with different K values
6. Evaluate performance metrics
7. Perform hyperparameter tuning using GridSearchCV
8. Build a Pipeline for automation
9. Predict heart disease on test data

---

## 🧠 Machine Learning Algorithm

### K-Nearest Neighbors (KNN)
KNN is a supervised Machine Learning classification algorithm that predicts output based on the nearest data points.

The project tests multiple K values:
- K = 3
- K = 5
- K = 7
- K = 9
---

## ⚖️ Feature Scaling

Since KNN is a distance-based algorithm, feature scaling is essential.

This project uses:

```python
StandardScaler()
```

to normalize feature values.

---

## 🔧 Hyperparameter Tuning

### GridSearchCV was used to optimize:

GridSearchCV is used to find the best K value using Cross Validation.

```python
param_grid = {"n_neighbors": [3,5,7,9]}
```

The model is also optimized using:

```python
scoring="recall"
```

to maximize Recall Score.

---

## 🔗 Pipeline Implementation

Pipeline is used to automate:
1. Feature Scaling
2. KNN Model Training

```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])
```

---

## 📈 Evaluation Metrics

The model is evaluated using:
- Accuracy Score
- Precision Score
- Recall Score

```python
print("Recall Score :", recall_score(y_test, y_pred))
print("Accuracy Score :", accuracy_score(y_test, y_pred))
print("Precision Score :", precision_score(y_test, y_pred))
```


---

## 📊 Sample Output

```python
Best k found by GridSearchCV: {'knn__n_neighbors': 7}

Recall Score : 0.90
Accuracy Score : 0.88
Precision Score : 0.87
```

---

## 📌 Key Concepts Covered

* K-Nearest Neighbors (KNN)
* Distance-based learning
* Feature scaling importance
* GridSearchCV for hyperparameter tuning
* Pipeline implementation in sklearn
* Model evaluation techniques

---

## 🚀 Future Improvements

* Try advanced models (Random Forest, XGBoost)
* Deploy using Flask or Streamlit web app
* Add real-time prediction system
* Add confusion matrix and ROC curve
* Improve feature engineering

---

## 👨‍💻 Author
Siddharth
---
