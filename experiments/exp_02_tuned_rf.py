# Hyperparameter tuning experiment

from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import (
    load_data,
    preprocess_data
)

from src.evaluate import evaluate_model

# Load dataset

df = load_data()

# Preprocess dataset

X_train, X_test, y_train, y_test = preprocess_data(df)

# Create improved model

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    random_state=42
)

# Train model

rf_model.fit(X_train, y_train)

# Evaluate model

evaluate_model(
    rf_model,
    X_test,
    y_test
)
