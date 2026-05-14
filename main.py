# Import required modules

import mlflow
import mlflow.sklearn

from src.preprocessing import (
    load_data,
    preprocess_data
)

from src.train import (
    train_random_forest,
    train_adaboost
)

from src.evaluate import evaluate_model

from src.utils import save_model

# Load dataset

df = load_data()

# Preprocess dataset

X_train, X_test, y_train, y_test = preprocess_data(df)

# Start MLflow experiment tracking

with mlflow.start_run():

    # Train Random Forest model

    rf_model = train_random_forest(
        X_train,
        y_train
    )

    # Evaluate Random Forest model

    accuracy, precision, recall, f1 = evaluate_model(
        rf_model,
        X_test,
        y_test
    )

    # Log metrics in MLflow

    mlflow.log_metric("accuracy", accuracy)

    mlflow.log_metric("precision", precision)

    mlflow.log_metric("recall", recall)

    mlflow.log_metric("f1_score", f1)

    # Log model

    mlflow.sklearn.log_model(
        rf_model,
        "random_forest_model"
    )

    # Save model

    save_model(
        rf_model,
        "models/random_forest.pkl"
    )

    print("Pipeline executed successfully")