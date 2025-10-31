1. Code setting up
1.1 Environment Description
Python version: Python 3.12.4
IDE: VSCode (1.86.2 (Universal))
OS: Darwin (MacOS) version='Darwin Kernel Version 23.2.0’
1.2 Library
You have to have all the libraries used in the project including: numpy,  pandas and sklearn
You can run these command in your terminal to install them:
python -m venv sklearn-env
source sklearn-env/bin/activate  # activate
pip install -U scikit-learn
pip install pandas

2. Update from project proposal
Cross-validation cv=4
- Imputation: Calculate cross_val_score() for Mean and Class-specific imputation + train them on 4 models and choose the best one for each model based on the highest F1 score.
- Normalization: Calculate cross_val_score() for MinMaxScaler() and StandardScaler() + train them on 4 models to choose the best one for each model based on the highest F1 score.
- Outlier removal: Calculate cross_val_score() for Local Outlier Factor (LOF) and Isolation forest + train them on 4 models and choose the best one for each model based on the highest F1 score.

3. Reproduction Instructions
How to run code from terminal: python s4869584.py - use this command to run the entire python file. This python file includes all coding for Pre-processing method comparison, Pre-processing, Hyperparameter Tuning Procedures, Model predicting and Report generation and because I did not separate the them so it will run every step from Pre-processing to Report generation (might take about 10-15 mins to finish). 

4. Final choice
The result after doing the comparison between imputation method (mean and class-specific):
- Class-specific Imputation Results: {'Decision Tree': {'mean_f1': 0.614, 'std_f1': 0.04}, 
                           'Random Forest': {'mean_f1': 0.655, 'std_f1': 0.032}, 
                           'k-NN': {'mean_f1': 0.074, 'std_f1': 0.024}, 
                           'GaussianNB': {'mean_f1': 0.217, 'std_f1': 0.001}}
- Mean Imputation Results: {'Decision Tree': {'mean_f1': 0.624, 'std_f1': 0.033}, 
                           'Random Forest': {'mean_f1': 0.675, 'std_f1': 0.039}, 
                           'k-NN': {'mean_f1': 0.074, 'std_f1': 0.024}, 
                           'GaussianNB': {'mean_f1': 0.217, 'std_f1': 0.001}}
Based on this result: Mean imputation will be applied for Decision tree, RandomForest tree, KNN and GNB. For KNN and GNB, even though the metrics do not show any significant changes but after training them with both imputation methods, it was seen to slightly increase the F1 score and accuracy with mean imputation so mean is also selected for KNN and GNB.

The result after doing the comparison between outlier removal methods (Local Outlier Factor and Isolation Forest):
- LOF Results: {'Decision Tree': {'mean_f1': 0.641, 'std_f1': 0.06}, 
               'Random Forest': {'mean_f1': 0.646, 'std_f1': 0.051}, 
               'k-NN': {'mean_f1': 0.073, 'std_f1': 0.024}, 
               'GaussianNB': {'mean_f1': 0.217, 'std_f1': 0.001}}
- Isolation Forest Results: {'Decision Tree': {'mean_f1': 0.628, 'std_f1': 0.04}, 
               'Random Forest': {'mean_f1': 0.725, 'std_f1': 0.049}, 
               'k-NN': {'mean_f1': 0.073, 'std_f1': 0.024}, 
               'GaussianNB': {'mean_f1': 0.217, 'std_f1': 0.001}}
Based on this result, Local Outlier Factor (LOF) is chosen for RandomForest, Decision tree, KNN and GNB. For KNN and GNB, even though the metrics do not show any significant changes but after training them with both imputation methods, it was seen to slightly increase the F1 score and accuracy with LOF so LOF  is also selected for KNN and GNB. 

The result after doing the comparison between normalization methods (with cv= 4):
- MinMaxScaler Results: {'Decision Tree': {'mean_f1': 0.626, 'std_f1': 0.041}, 
                        'Random Forest': {'mean_f1': 0.683, 'std_f1': 0.039}, 
                        'k-NN': {'mean_f1': 0.752, 'std_f1': 0.043}, 
                        'GaussianNB': {'mean_f1': 0.217, 'std_f1': 0.001}}
- StandardScaler Results: {'Decision Tree': {'mean_f1': 0.602, 'std_f1': 0.009}, 
                        'Random Forest': {'mean_f1': 0.666, 'std_f1': 0.076}, 
                        'k-NN': {'mean_f1': 0.729, 'std_f1': 0.031}, 
                        'GaussianNB': {'mean_f1': 0.217, 'std_f1': 0.001}}
Based on this result, StandardScaler is chosen for Decision tree and RandomForest tree and MinMaxScaler is chosen for KNN and GNB.

After training both 4 models, we have the below result:
- Decision tree:
	Tuning params:
	dt_parameters = {
        'max_depth': [3, 5],  
        'criterion': ('gini', 'entropy'),
        'max_features': ('auto', 'sqrt', 'log2'),
        'min_samples_split': (10, 20, 50),
        'min_samples_leaf': (1, 2)
    }
DecisionTree best param:  {'criterion': 'gini', 'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 50}
Accuracy score:  0.917
F1 score:  0.56
F1 score macro:  0.757

- RandomForest tree:
	Tuning params:
	   rf_params = {
        "n_estimators": [100, 200, 300],
        "max_depth": (3, 5, 7, 10),
        "min_samples_split": [2, 6, 10],
        "min_samples_leaf": [1, 3, 4],
    }	
RandomForestTree best param:  {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
Accuracy score:  0.938
F1 score:  0.675
F1 score macro:  0.82

- KNN:
	Tuning params:
	knn_params = {
        "n_neighbors": range(1, 30, 2),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
        "leaf_size": range(1, 50, 5)
    }
KNN best param:  {'leaf_size': 1, 'metric': 'manhattan', 'n_neighbors': 9, 'weights': 'uniform'}
Accuracy score:  0.953
F1 score:  0.759
F1 score macro:  0.867

- GNB:
	Tuning params:
gnb_param_grid = {
        "var_smoothing": np.logspace(0, -9, num=100)
    }
Naive Bayes best param:  {'var_smoothing': 1.0}
Accuracy score:  0.935
F1 score:  0.705
F1 score macro:  0.834

Based on this result, KNN is the final choice to be used to predict the test_data.csv with the highest F1 score.
