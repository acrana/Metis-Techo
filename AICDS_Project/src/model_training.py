# src/model_training.py

import pandas as pd
import numpy as np
from preprocessing import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import joblib
import os
import json

def train_model():
    # Load and preprocess data
    data = preprocess_data()

    # Separate features and target
    X = data.drop(['Had_ADE'], axis=1)
    y = data['Had_ADE']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    scaler_path = os.path.join(models_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # Build the model
    input_dim = X_train_scaled.shape[1]
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=8, validation_data=(X_test_scaled, y_test))

    # Save the model
    model_path = os.path.join(models_dir, 'model.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Save feature columns
    feature_columns = list(X.columns)
    feature_columns_path = os.path.join(models_dir, 'features.json')
    with open(feature_columns_path, 'w') as f:
        json.dump(feature_columns, f)
    print(f"Feature columns saved to {feature_columns_path}")

if __name__ == '__main__':
    train_model()
