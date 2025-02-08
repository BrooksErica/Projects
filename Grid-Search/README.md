Hyperparameter Tuning and Model Evaluation using Cross-Validation
Overview
This project implements hyperparameter tuning and model evaluation using cross-validation on multiple classifiers. The models tested include:
•	Random Forest Classifier
•	Logistic Regression
•	Support Vector Classifier (SVC)
Dataset
The dataset used in this project is the Wine Quality Dataset, specifically winequality-white.csv, which contains various chemical attributes of white wine samples along with their quality ratings.
Data Preprocessing
•	The dataset is loaded using numpy.genfromtxt().
•	Features (M) and labels (L) are extracted from the dataset.
•	Labels represent wine quality scores.
Model Selection and Hyperparameter Tuning
1. Cross-Validation Strategy
•	K-Fold Cross-Validation (k=5) is used to evaluate model performance.
•	The dataset is split into k folds, and models are trained and tested across different folds.
2. Classifiers and Hyperparameters
A dictionary (all_params) is defined to store classifier types and their hyperparameter search spaces.
Random Forest Classifier
•	n_estimators: [10, 50, 100]
•	max_depth: [5, 10, None]
Logistic Regression
•	C: [1.0, 0.1, 0.01]
•	max_iter: [100, 200, 300]
Support Vector Classifier (SVC)
•	C: [1.0, 0.01, 0.1]
•	kernel: ['rbf', 'poly', 'sigmoid']
Functionality
1. gen_clf_models() Function
•	Iterates through classifiers and their hyperparameters.
•	Performs K-Fold Cross-Validation to evaluate each model.
•	Uses scikit-learn's ParameterGrid to generate all possible hyperparameter combinations.
•	Computes evaluation metrics:
o	Accuracy
o	Precision
o	Recall
o	F1-Score
•	Stores results for each classifier and hyperparameter combination.
2. Training & Evaluation
•	The function returns a dictionary containing model performances across all hyperparameter settings.
•	Results are printed in a structured format.
![image](https://github.com/user-attachments/assets/ea8c24ea-222c-4058-8cf2-4d72d6c40f55)
