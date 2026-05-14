# Import models

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier
)

# Import configuration values

from src.config import (
    RF_ESTIMATORS,
    RF_MAX_DEPTH,
    AB_ESTIMATORS,
    RANDOM_STATE
)

# Train Random Forest model

def train_random_forest(X_train, y_train):

    rf_model = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )

    rf_model.fit(X_train, y_train)

    return rf_model

# Train AdaBoost model

def train_adaboost(X_train, y_train):

    ab_model = AdaBoostClassifier(
        n_estimators=AB_ESTIMATORS,
        random_state=RANDOM_STATE
    )

    ab_model.fit(X_train, y_train)

    return ab_model