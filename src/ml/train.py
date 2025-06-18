import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import joblib

def train_model(merged_df, use_randomized_search=True):
    # Define Features (X) and Target (y)
    features = ["name ID", "Hazard Class Encoded", "Lab Number", "Group Encoded"]
    X = pd.get_dummies(merged_df[features], drop_first=True)
    y = merged_df["Storage ID"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

    # Define hyperparameter grid or distributions
    param_dist = {
        'n_estimators': [300, 500, 1000],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'max_features': ['sqrt', 'log2', None]
    }

    # Choose between GridSearchCV and RandomizedSearchCV
    if use_randomized_search:
        print("Using RandomizedSearchCV for hyperparameter tuning...")
        search = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=42),
                                    param_distributions=param_dist,
                                    n_iter=50,  # Number of random combinations to try
                                    cv=2,  # 2-fold cross-validation
                                    scoring='accuracy',
                                    random_state=42,
                                    n_jobs=-1)  # Use all available CPU cores
    else:
        print("Using GridSearchCV for hyperparameter tuning...")
        search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                              param_grid=param_dist,
                              cv=2,  # 2-fold cross-validation
                              scoring='accuracy',
                              n_jobs=-1)  # Use all available CPU cores

    # Perform hyperparameter search
    search.fit(X_train, y_train)

    # # Print the best parameters and the corresponding score
    print("\n\nBest Hyperparameters:", search.best_params_)
    print("Best CV Accuracy:", search.best_score_)

    # Retrieve the best model
    best_rf_model = search.best_estimator_

    # Train the model on the full training set with the best parameters
    best_rf_model.fit(X_train, y_train)
    print("\n\nOptimized Random Forest model trained successfully!")



    # Save the trained model
    joblib.dump(best_rf_model, "../models/random_forest.joblib")
    print("\nModel saved to '../models/random_forest.joblib'")

    return best_rf_model, X_test, y_test
