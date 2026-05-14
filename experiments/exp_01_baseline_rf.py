# Baseline Random Forest Experiment

from src.preprocessing import (
    load_data,
    preprocess_data
)

from src.train import train_random_forest

from src.evaluate import evaluate_model

# Load dataset

df = load_data()

# Preprocess data

X_train, X_test, y_train, y_test = preprocess_data(df)

# Train model

rf_model = train_random_forest(
    X_train,
    y_train
)

# Evaluate model

evaluate_model(
    rf_model,
    X_test,
    y_test
)