# predictions.py

import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_model(model_path):
    """Load the trained model."""
    return tf.keras.models.load_model(model_path)

def load_data(file_path):
    """Load new data for predictions."""
    return pd.read_csv(file_path)

def make_predictions(model, data):
    """Make predictions using the trained model."""
    predictions = model.predict(data)
    return predictions

if __name__ == "__main__":
    model_path = "models/cdss_model.h5"
    data_path = "data/new_simpledata.csv"  # New data for prediction

    # Load model and data
    model = load_model(model_path)
    data = load_data(data_path)

    # Assuming the new data needs to be scaled as well
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Make predictions
    predictions = make_predictions(model, data)

    # Print predictions
    print("Predictions:")
    print(predictions)

    print("Prediction process completed successfully.")
