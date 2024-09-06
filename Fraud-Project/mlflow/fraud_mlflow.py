import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
from mlflow.models.signature import infer_signature
import mlflow
from geopy.distance import great_circle
from xgboost import XGBClassifier

if __name__ == "__main__":
    # MLflow experiment
    experiment_name = "fraud_test"

    mlflow.set_tracking_uri("https://mlflow-lead-a5e0197c9b59.herokuapp.com/")
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Download CSV
    df_fraud = pd.read_csv("fraudTest.csv")

    # Distance merch - transaction
    def calculate_distance(row):
        try:
            coord1 = (row['lat'], row['long'])
            coord2 = (row['merch_lat'], row['merch_long'])
            return great_circle(coord1, coord2).kilometers
        except:
            return None
        
    df_fraud['distance'] = df_fraud.apply(calculate_distance, axis=1)

    # Select columns to drop from CSV
    columns_to_drop = ["Unnamed: 0", "trans_date_trans_time", "cc_num", "merchant", "job", "first", "last", "street", "unix_time"]
    data = df_fraud.drop(columns=columns_to_drop)

    # Features and target
    X = df_fraud.drop(columns=['is_fraud'])
    y = df_fraud['is_fraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Num and cat features
    categorical_features = ['category', 'gender', 'city', 'state', 'zip', 'dob', 'trans_num']
    numeric_features = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'distance']

    # Create transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Create pipeline with preprocessor
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Logistic Regression model and metrics
    with mlflow.start_run() as run:
        LR_model = LogisticRegression()
        LR_model.fit(X_train_transformed, y_train)
        
        # Predict
        y_train_pred = LR_model.predict(X_train_transformed)
        y_test_pred = LR_model.predict(X_test_transformed)

        # Log Logistic Regression model
        mlflow.sklearn.log_model(
            sk_model=LR_model,
            artifact_path="fraud_lr_model",
            signature=infer_signature(X_train_transformed, y_train_pred)
        )   

        # Training metrics
        mlflow.log_metric("training_accuracy_lr", accuracy_score(y_train, y_train_pred))
        mlflow.log_metric("training_precision_lr", precision_score(y_train, y_train_pred))
        mlflow.log_metric("training_recall_lr", recall_score(y_train, y_train_pred))
        mlflow.log_metric("training_f1_score_lr", f1_score(y_train, y_train_pred))

        # Test metrics
        mlflow.log_metric("testing_accuracy_lr", accuracy_score(y_test, y_test_pred))
        mlflow.log_metric("testing_precision_lr", precision_score(y_test, y_test_pred))
        mlflow.log_metric("testing_recall_lr", recall_score(y_test, y_test_pred))
        mlflow.log_metric("testing_f1_score_lr", f1_score(y_test, y_test_pred))

    # XGBoost model without hyperparameter tuning
    with mlflow.start_run() as run:
        # Initialize the XGBoost model with default parameters
        model = XGBClassifier(random_state=42)
        
        # Train the model
        model.fit(X_train_transformed, y_train)
        
        # Predict on the test set
        y_test_pred = model.predict(X_test_transformed)

        # Log XGBoost model
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="fraud_xgb_model",
            signature=infer_signature(X_train_transformed, y_test_pred)
        )

        # Metrics
        print("Classification Report (Test Data):")
        print(classification_report(y_test, y_test_pred))
        print("Accuracy:", accuracy_score(y_test, y_test_pred))
        
        # Log metrics for XGBoost
        mlflow.log_metric("testing_accuracy_xgb", accuracy_score(y_test, y_test_pred))
        mlflow.log_metric("testing_precision_xgb", precision_score(y_test, y_test_pred))
        mlflow.log_metric("testing_recall_xgb", recall_score(y_test, y_test_pred))
        mlflow.log_metric("testing_f1_score_xgb", f1_score(y_test, y_test_pred))
