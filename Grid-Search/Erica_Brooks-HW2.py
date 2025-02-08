#import libraries
import numpy as np
from sklearn.model_selection import ParameterGrid, KFold
#import metrics to evaluate model
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
#import classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
#cross - validation
from sklearn.model_selection import KFold

#Part 1: Function of classifiers 

def gen_clf_models(all_params, data, n_splits=5):
     
     if len(data) == 2:
         M, L = data
         n_folds = n_splits
     else:
         M, L, n_folds = data

     kf = KFold(n_splits=n_folds)  # Establish cross-validation
     results = {}  # Store all results

     for key, value in all_params.items():
        clf_class = value['clf']  # Extract the classifier class
        hyper_params = value['hypers']  # Extract the hyperparameters
        
        # Generate all combinations of hyperparameters using ParameterGrid
        param_grid = list(ParameterGrid(hyper_params))
        
        # Create a model instance for each hyperparameter combination
        model_results = []

        for params in param_grid:
            fold_metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []}
            
            # Perform KFold cross-validation
            for train_index, test_index in kf.split(M):
                X_train, X_test = M[train_index], M[test_index]
                y_train, y_test = L[train_index], L[test_index]

                model = clf_class(**params)  # Instantiate the model with params
                model.fit(X_train, y_train)  # Store (key, model, params) tuple

                # Predict and calculate metrics
                y_pred = model.predict(X_test)
                fold_metrics['Accuracy'].append(accuracy_score(y_test, y_pred))
                fold_metrics['Precision'].append(precision_score(y_test, y_pred, average='weighted', zero_division = 1))
                fold_metrics['Recall'].append(recall_score(y_test, y_pred, average='weighted'))
                fold_metrics['F1-Score'].append(f1_score(y_test, y_pred, average='weighted'))

                # Compute the average metrics over all folds
                avg_metrics = {metric: np.mean(scores) for metric, scores in fold_metrics.items()}

                # Store hyperparameters and metrics for this combination
                model_results.append({'hyperparameters': params, 'metrics': avg_metrics})

        # Add results for this model to the main dictionary
        results[key] = model_results

     return results

#Part 2: Expand the number of classifiers and hyper parameters

all_params = {'rf':{'clf':RandomForestClassifier,
                    'hypers': {
                              'n_estimators':[10,50,100],
                              'max_depth':[5,10,None]}},
                              
            'log_reg':{'clf':LogisticRegression,
                    'hypers': {
                              'C':[1.0,0.1,0.01],
                              'max_iter':[100,200,300]}},

            'svc':{'clf':SVC,
                    'hypers': {
                              'C':[1.0,0.01,0.1],
                              'kernel':['rbf','poly','sigmoid']}}
}

#sample data
data = np.genfromtxt('wine+quality/winequality-white.csv', delimiter=';', dtype=float, encoding='utf-8', skip_header=1)

M = data[:, :-1]
L = data[:, -1]

print(M)

#Generate models
results = gen_clf_models(all_params, (M,L), 5)

for model_name, model_results in results.items():
    print(f"Model: {model_name}")
    for result in model_results:
        print(f"  Hyperparameters: {result['hyperparameters']}, Metrics: {result['metrics']}")