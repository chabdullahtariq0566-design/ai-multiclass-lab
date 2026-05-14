# Import required libraries

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PATH,
    TEST_SIZE,
    RANDOM_STATE
)

# Load dataset

def load_data():

    df = pd.read_csv(DATA_PATH)

    return df

# Preprocess dataset

def preprocess_data(df):

    # Encode target labels

    encoder = LabelEncoder()

    df['Class'] = encoder.fit_transform(
        df['Class']
    )

    # Separate features and labels

    X = df.drop('Class', axis=1)

    y = df['Class']

    # Train-test split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test