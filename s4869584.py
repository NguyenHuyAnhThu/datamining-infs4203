import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def reportGenerate(model, accuracy, f1):
    test_data = pd.read_csv("test_data.csv")

    # Predict target column using the model
    predicted_target = model.predict(test_data)

    # Write predictions to .infs file
    filename = f"s4869584.infs4203_final"
    with open(filename, "w") as f:
        for prediction in predicted_target:
            f.write(str(prediction) + ",\n")  # Add comma and newline
        # Append accuracy and F1 scores (if calculated)
        f.write("{:.4f}".format(accuracy))
        f.write(", {:.4f}".format(f1))

    print("Predictions and scores saved to s4869584.infs4203!")


def main():
    df = pd.read_csv('DM_Project_24.csv')

    df_impu_1_103 = df.copy()
    df_impu_1_103.iloc[:,:103] = df_impu_1_103.iloc[:,:103].fillna(df_impu_1_103.iloc[:,:103].mean())

    df_impu_104_105 = df.copy()
    df_impu_104_105.iloc[:, [103, 104]] = df_impu_104_105.iloc[:, [103, 104]].fillna(df_impu_104_105.iloc[:, [103, 104]].mode().iloc[0])

    df_impu_all = pd.concat([
        df_impu_1_103.iloc[:, :103],
        df_impu_104_105.iloc[:, 103:106]
    ], axis=1)

    # Split the "target" column
    y = df_impu_all.iloc[:,-1].values # get"Target" colum
    X = df_impu_all.iloc[:,:-1].values # get column 1-105
    X_num = df_impu_all.iloc[:,:103].values #numerical features
    X_nom = df_impu_all.iloc[:,103:-1].values #nominal features
    
    # IMPUTATION/OUTLIER/NORMALIZATION METHOD COMPARISION
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'k-NN': KNeighborsClassifier(),
        'GaussianNB': GaussianNB()
    }

    def evaluate_model(model, X_train, y_train):
        scores = cross_val_score(model, X_train, y_train, cv=4, scoring='f1')
        return np.mean(scores), np.std(scores)
    
    # 1. IMPUTATION METHOD COMPARISION
    # Mean Imputation
    mean_imputer = SimpleImputer(strategy='mean')
    X_train_mean = mean_imputer.fit_transform(X_train)
    X_test_mean = mean_imputer.transform(X_test)

    # Class-Specific Imputation Function
    def class_specific_imputation(X, y, strategy='mean'):
        X_imputed = X.copy()
        for cls in np.unique(y):
            # Impute missing values for each class separately
            class_indices = np.where(y == cls)[0]
            imputer = SimpleImputer(strategy=strategy)
            X_imputed[class_indices] = imputer.fit_transform(X[class_indices])
        return X_imputed

    # Apply class-specific imputation to the training data
    X_train_class_specific = class_specific_imputation(X_train, y_train)
    X_test_class_specific = class_specific_imputation(X_test, y_test)
    
    # Evaluate each model after mean imputation
    mean_imputation_results = {}
    for name, model in models.items():
        mean_score, std_score = evaluate_model(model, X_train_mean, y_train)
        mean_imputation_results[name] = {'mean_f1': round(mean_score, 3), 'std_f1': round(std_score, 3)}

    print("Mean Imputation Results:", mean_imputation_results)

    class_specific_results = {}
    for name, model in models.items():
        mean_score, std_score = evaluate_model(model, X_train_class_specific, y_train)
        class_specific_results[name] = {'mean_f1': round(mean_score, 3), 'std_f1': round(std_score, 3)}

    print("Class-Specific Imputation Results:", class_specific_results)

    # 2. OUTLIER METHOD COMPARISION
    # LOF
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
    lof_outliers = lof.fit_predict(X_train)
    X_train_lof = X_train[lof_outliers == 1]  # Keep only non-outliers
    y_train_lof = y_train[lof_outliers == 1]

    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    iso_outliers = iso_forest.fit_predict(X_train)
    X_train_iso = X_train[iso_outliers == 1]  # Keep only non-outliers
    y_train_iso = y_train[iso_outliers == 1]

    # Evaluate each model after LOF outlier removal
    lof_results = {}
    for name, model in models.items():
        mean_score, std_score = evaluate_model(model, X_train_lof, y_train_lof)
        lof_results[name] = {'mean_f1': round(mean_score, 3), 'std_f1': round(std_score, 3)}

    print("LOF Results:", lof_results)

    iso_results = {}
    for name, model in models.items():
        mean_score, std_score = evaluate_model(model, X_train_iso, y_train_iso)
        iso_results[name] = {'mean_f1': round(mean_score, 3), 'std_f1': round(std_score, 3)}

    print("Isolation Forest Results:", iso_results)

    # 3. NORMALIZATION METHOD COMPARISION
    # Apply MinMaxScaler and StandardScaler
    minmax_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()

    X_train_minmax = minmax_scaler.fit_transform(X_train)
    X_test_minmax = minmax_scaler.transform(X_test)

    X_train_standard = standard_scaler.fit_transform(X_train)
    X_test_standard = standard_scaler.transform(X_test)

    # Evaluate each model with MinMaxScaler
    minmax_results = {}
    for name, model in models.items():
        mean_score, std_score = evaluate_model(model, X_train_minmax, y_train)
        minmax_results[name] = {'mean_f1': round(mean_score, 3), 'std_f1': round(std_score, 3)}

    print("MinMaxScaler Results:", minmax_results)
    # Train models and evaluate them with StandardScaler
    standard_results = {}
    for name, model in models.items():
        mean_score, std_score = evaluate_model(model, X_train_standard, y_train)
        standard_results[name] = {'mean_f1': round(mean_score, 3), 'std_f1': round(std_score, 3)}

    print("StandardScaler Results:", standard_results)
    
    # DECISION TREE
    # Isolation forest removal
    X_train, X_test = train_test_split(df_impu_all, test_size=0.25, random_state=0, stratify=y)
    
    model = LocalOutlierFactor(n_neighbors=20,contamination=0.01)  
    model.fit(X_train)
    y_pred_train = model.fit_predict(X_train)

    inliers_train = X_train[y_pred_train == 1]
    X_train_cleaned = inliers_train
    
    # Normalization
    X_num_train = X_train_cleaned.iloc[:, :103] 
    X_nom_train = X_train_cleaned.iloc[:, 103:-1] 

    X_num_test = X_test.iloc[:, :103]  
    X_nom_test = X_test.iloc[:, 103:-1]  

    y_train = X_train_cleaned.iloc[:, -1]
    y_test = X_test.iloc[:, -1]

    scaler = StandardScaler()
    scaler.fit(X_num_train)
    X_num_train = scaler.transform(X_num_train)
    X_num_test = scaler.transform(X_num_test)

    X_train = np.concatenate((X_num_train, X_nom_train), axis=1)
    X_test = np.concatenate((X_num_test, X_nom_test), axis=1)
    
    # Build tree
    dt_parameters = {
        'max_depth': [3, 5],  
        'criterion': ('gini', 'entropy'),
        'max_features': ('auto', 'sqrt', 'log2'),
        'min_samples_split': (10, 20, 50),
        'min_samples_leaf': (1, 2)
    }

    dt=DecisionTreeClassifier(random_state=101)

    dt_Tunning_grid = GridSearchCV(estimator=dt, param_grid=dt_parameters, n_jobs=1, cv=4, scoring="f1", error_score=0)
    dt_grid_results = dt_Tunning_grid.fit(X_train, y_train)
    print("DecisionTree best param: ", dt_grid_results.best_params_)

    dt_final_model = dt.set_params(**dt_grid_results.best_params_)
    dt_final_model.fit(X_train, y_train)
    dt_y_pred = dt_final_model.predict(X_test)

    print("Accuracy score: ", round(accuracy_score(y_test, dt_y_pred),3))
    print("F1 score: ", round(f1_score(y_test, dt_y_pred),3))
    print ("F1 score macro: ", round(f1_score(y_test, dt_y_pred, average='macro'),3))
    
    # reportGenerate("Decision tree", dt_final_model, accuracy_score(y_test, dt_y_pred), f1_score(y_test, dt_y_pred))
    
    # RANDOM FOREST TREE
    # LocalOutlierFactor removal
    X_train, X_test = train_test_split(df_impu_all, test_size=0.25, random_state=0, stratify=y)
    
    model = LocalOutlierFactor(n_neighbors=20, contamination=0.01) 
    model.fit(X_train)
    y_pred_train = model.fit_predict(X_train)

    inliers_train = X_train[y_pred_train == 1]
    X_train_cleaned = inliers_train
    
    # Normalization
    X_num_train = X_train_cleaned.iloc[:, :103]  
    X_nom_train = X_train_cleaned.iloc[:, 103:-1] 

    X_num_test = X_test.iloc[:, :103]  
    X_nom_test = X_test.iloc[:, 103:-1]  

    y_train = X_train_cleaned.iloc[:, -1]
    y_test = X_test.iloc[:, -1]

    scaler = StandardScaler()
    scaler.fit(X_num_train)
    X_num_train = scaler.transform(X_num_train)
    X_num_test = scaler.transform(X_num_test)

    X_train = np.concatenate((X_num_train, X_nom_train), axis=1)
    X_test = np.concatenate((X_num_test, X_nom_test), axis=1)   
    
    # Build tree
    rf = RandomForestClassifier(random_state=101)
    rf_params = {
        "n_estimators": [100, 200, 300],
        "max_depth": (3, 5, 7, 10),
        "min_samples_split": [2, 6, 10],
        "min_samples_leaf": [1, 3, 4],
    }

    rf_grid_search = GridSearchCV(estimator=rf, param_grid=rf_params, n_jobs=1, cv=4, scoring="f1", error_score=0)
    rf_grid_results = rf_grid_search.fit(X_train, y_train)
    print("RandomForestTree best param: ", rf_grid_results.best_params_)

    rf_final_model = rf.set_params(**rf_grid_results.best_params_)
    rf_final_model.fit(X_train, y_train)

    rf_y_pred = rf_final_model.predict(X_test)
    
    print("Accuracy score: ", round(accuracy_score(y_test, rf_y_pred),3))
    print("F1 score: ", round(f1_score(y_test, rf_y_pred),3))
    print("F1 score macro: ", round(f1_score(y_test, rf_y_pred, average='macro'),3))
    
    # reportGenerate("Random Forest", rf_final_model, accuracy_score(y_test, rf_y_pred), f1_score(y_test, rf_y_pred))
    
    # KNN
    # # MinMaxScaler for numerical features
    # X_num_train, X_num_test, X_nom_train, X_nom_test, y_train, y_test = train_test_split(X_num, X_nom, y, random_state = None, stratify=y)

    # scaler = MinMaxScaler()
    # scaler.fit(X_num_train)  # Fit the scaler on training data

    # # Apply MinMaxScaler to training and testing sets (using training set statistics)
    # X_num_train_scaled = scaler.transform(X_num_train)
    # X_num_test_scaled = scaler.transform(X_num_test)  # Use training set min and max

    # # Combine scaled numerical features with nominal features
    # X_train = np.concatenate((X_num_train_scaled, X_nom_train), axis=1)
    # X_test = np.concatenate((X_num_test_scaled, X_nom_test), axis=1)
    
    # LocalOutlierFactor removal
    X_train, X_test = train_test_split(df_impu_all, test_size=0.25, random_state=0, stratify=y)
    
    model = IsolationForest(contamination=0.01)  
    model.fit(X_train)
    y_pred_train = model.predict(X_train)

    inliers_train = X_train[y_pred_train == 1]
    X_train_cleaned = inliers_train
    
    # Normalization
    X_num_train = X_train_cleaned.iloc[:, :103]  
    X_nom_train = X_train_cleaned.iloc[:, 103:-1]  

    X_num_test = X_test.iloc[:, :103] 
    X_nom_test = X_test.iloc[:, 103:-1] 

    y_train = X_train_cleaned.iloc[:, -1]
    y_test = X_test.iloc[:, -1]

    scaler = MinMaxScaler()
    scaler.fit(X_num_train)
    X_num_train = scaler.transform(X_num_train)
    X_num_test = scaler.transform(X_num_test)

    X_train = np.concatenate((X_num_train, X_nom_train), axis=1)
    X_test = np.concatenate((X_num_test, X_nom_test), axis=1)

    # Train model
    knn = KNeighborsClassifier()
    knn_params = {
        "n_neighbors": range(1, 30, 2),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
        "leaf_size": range(1, 50, 5)
    }

    knn_grid_search = GridSearchCV(estimator=knn, param_grid=knn_params, n_jobs=1, cv=4, scoring="f1", error_score=0)
    knn_grid_results = knn_grid_search.fit(X_train, y_train)
    print("KNN best param: ", knn_grid_results.best_params_)

    knn_final_model = knn.set_params(**knn_grid_results.best_params_)
    knn_final_model.fit(X_train, y_train)

    knn_y_pred = knn_final_model.predict(X_test)
    
    print("Accuracy score: ", round(accuracy_score(y_test, knn_y_pred), 3))
    print("F1 score: ", round(f1_score(y_test, knn_y_pred), 3))
    print("F1 score macro: ", round(f1_score(y_test, knn_y_pred, average='macro'), 3))
    
    reportGenerate(knn_final_model, round(accuracy_score(y_test, knn_y_pred), 3), round(f1_score(y_test, knn_y_pred), 3))
    
    # GNB
    # # MinMaxScaler for numerical features
    # X_num_train, X_num_test, X_nom_train, X_nom_test, y_train, y_test = train_test_split(X_num, X_nom, y, random_state = None, stratify=y)

    # scaler = MinMaxScaler()
    # scaler.fit(X_num_train)
    # X_num_train_scaled = scaler.transform(X_num_train)
    # X_num_test_scaled = scaler.transform(X_num_test) 
    # X_train = np.concatenate((X_num_train_scaled, X_nom_train), axis=1)
    # X_test = np.concatenate((X_num_test_scaled, X_nom_test), axis=1)
    
    # LocalOutlierFactor removal
    X_train, X_test = train_test_split(df_impu_all, test_size=0.25, random_state=0, stratify=y)
    
    model = IsolationForest(contamination=0.01)  
    model.fit(X_train)
    y_pred_train = model.predict(X_train)

    inliers_train = X_train[y_pred_train == 1]
    X_train_cleaned = inliers_train
    
    # Normalization
    X_num_train = X_train_cleaned.iloc[:, :103]  
    X_nom_train = X_train_cleaned.iloc[:, 103:-1]  
    X_num_test = X_test.iloc[:, :103] 
    X_nom_test = X_test.iloc[:, 103:-1]  

    y_train = X_train_cleaned.iloc[:, -1]
    y_test = X_test.iloc[:, -1]

    scaler = MinMaxScaler()
    scaler.fit(X_num_train)
    X_num_train = scaler.transform(X_num_train)
    X_num_test = scaler.transform(X_num_test)

    X_train = np.concatenate((X_num_train, X_nom_train), axis=1)
    X_test = np.concatenate((X_num_test, X_nom_test), axis=1)

    # Train model
    gnb = GaussianNB()

    gnb_param_grid = {
        "var_smoothing": np.logspace(0, -9, num=100)
    }

    gnb_search = GridSearchCV(gnb, gnb_param_grid, scoring="f1", cv=4)
    gnb_grid_results = gnb_search.fit(X_train, y_train)
    print("Naive Bayes best param: ", gnb_grid_results.best_params_)

    gnb_final_model = gnb.set_params(**gnb_grid_results.best_params_)
    gnb_final_model.fit(X_train, y_train)

    gnb_y_pred = gnb_final_model.predict(X_test)

    print("Accuracy score: ", round(accuracy_score(y_test, gnb_y_pred),3))
    print("F1 score: ", round(f1_score(y_test, gnb_y_pred),3))
    print("F1 score macro: ", round(f1_score(y_test, gnb_y_pred, average='macro'),3))
    
    # reportGenerate("GNB", gnb_final_model, accuracy_score(y_test, gnb_y_pred), f1_score(y_test, gnb_y_pred))
        
if __name__ == "__main__":
    main()





