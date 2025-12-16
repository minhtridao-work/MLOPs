from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
import numpy as np
# import pandas as pd  # Not required for this project

def generate_data(n_samples=1000, n_features=10, augment=False):
    """Generate synthetic classification data"""
    print(f"Generating data with {n_samples} samples and {n_features} features...")
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=8,
        n_redundant=2,
        random_state=42
    )
    
    if augment:
        print("Applying data augmentation (noise and scaling)...")
        # Add noise
        noise = np.random.normal(0, 0.1, X.shape)
        X = X + noise
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Save scaler for later use
        joblib.dump(scaler, 'scaler.pkl')
    
    return X, y

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a model"""
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, accuracy

def hyperparameter_tuning(model_class, param_grid, X_train, y_train, model_name):
    """Perform hyperparameter tuning using GridSearchCV"""
    print(f"\nPerforming hyperparameter tuning for {model_name}...")
    
    grid_search = GridSearchCV(
        estimator=model_class,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def main():
    print("=== MLOps Classification Project ===")
    
    # Step 1: Generate and prepare data
    X, y = generate_data(n_samples=2000, augment=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Dictionary to store model results
    model_results = {}
    
    # Step 2: Train simple Logistic Regression
    lr_model, lr_accuracy = train_and_evaluate_model(
        LogisticRegression(random_state=42),
        X_train, X_test, y_train, y_test,
        "Logistic Regression"
    )
    model_results['Logistic Regression'] = {
        'model': lr_model,
        'accuracy': lr_accuracy
    }
    
    # Step 3: Hyperparameter tuning for Logistic Regression
    lr_param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [100, 200]
    }
    
    lr_tuned = hyperparameter_tuning(
        LogisticRegression(random_state=42),
        lr_param_grid,
        X_train, y_train,
        "Logistic Regression"
    )
    
    lr_tuned_model, lr_tuned_accuracy = train_and_evaluate_model(
        lr_tuned, X_train, X_test, y_train, y_test,
        "Logistic Regression (Tuned)"
    )
    model_results['Logistic Regression (Tuned)'] = {
        'model': lr_tuned_model,
        'accuracy': lr_tuned_accuracy
    }
    
    # Step 4: Train Random Forest
    rf_model, rf_accuracy = train_and_evaluate_model(
        RandomForestClassifier(random_state=42, n_estimators=100),
        X_train, X_test, y_train, y_test,
        "Random Forest"
    )
    model_results['Random Forest'] = {
        'model': rf_model,
        'accuracy': rf_accuracy
    }
    
    # Step 5: Hyperparameter tuning for Random Forest
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    
    rf_tuned = hyperparameter_tuning(
        RandomForestClassifier(random_state=42),
        rf_param_grid,
        X_train, y_train,
        "Random Forest"
    )
    
    rf_tuned_model, rf_tuned_accuracy = train_and_evaluate_model(
        rf_tuned, X_train, X_test, y_train, y_test,
        "Random Forest (Tuned)"
    )
    model_results['Random Forest (Tuned)'] = {
        'model': rf_tuned_model,
        'accuracy': rf_tuned_accuracy
    }
    
    # Step 6: Train XGBoost
    xgb_model, xgb_accuracy = train_and_evaluate_model(
        XGBClassifier(random_state=42, eval_metric='logloss'),
        X_train, X_test, y_train, y_test,
        "XGBoost"
    )
    model_results['XGBoost'] = {
        'model': xgb_model,
        'accuracy': xgb_accuracy
    }
    
    # Step 7: Hyperparameter tuning for XGBoost
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    
    xgb_tuned = hyperparameter_tuning(
        XGBClassifier(random_state=42, eval_metric='logloss'),
        xgb_param_grid,
        X_train, y_train,
        "XGBoost"
    )
    
    xgb_tuned_model, xgb_tuned_accuracy = train_and_evaluate_model(
        xgb_tuned, X_train, X_test, y_train, y_test,
        "XGBoost (Tuned)"
    )
    model_results['XGBoost (Tuned)'] = {
        'model': xgb_tuned_model,
        'accuracy': xgb_tuned_accuracy
    }
    
    # Step 8: Compare results and find best model
    print("\n=== Model Comparison ===")
    best_model_name = None
    best_accuracy = 0
    
    for name, result in model_results.items():
        print(f"{name}: {result['accuracy']:.4f}")
        if result['accuracy'] > best_accuracy:
            best_accuracy = result['accuracy']
            best_model_name = name
    
    print(f"\nBest Model: {best_model_name} with accuracy: {best_accuracy:.4f}")
    
    # Step 9: Save best model to model registry
    best_model = model_results[best_model_name]['model']
    model_path = "best_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"Best model saved to: {model_path}")
    
    # Save model metadata
    metadata = {
        'model_name': best_model_name,
        'accuracy': best_accuracy,
        'model_path': model_path,
        'all_results': {name: result['accuracy'] for name, result in model_results.items()}
    }
    
    import json
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Model metadata saved to: model_metadata.json")
    
    return best_model, best_accuracy

if __name__ == "__main__":
    main()