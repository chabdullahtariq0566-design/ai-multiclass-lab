# Import joblib for saving models

import joblib

# Save trained model

def save_model(model, path):

    joblib.dump(model, path)

    print(f"Model saved at {path}")