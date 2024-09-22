import os
from preprocessing import preprocess_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import joblib

def train_model():
    # Load and preprocess data
    data = preprocess_data()

    # Separate features and target
    X = data.drop(['Had_ADE'], axis=1)
    y = data['Had_ADE']

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    scaler_path = os.path.join(models_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # Build the model
    input_dim = X_scaled.shape[1]
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
    model.fit(X_scaled, y, epochs=50, batch_size=8, validation_split=0.2)

    # Save the model
    model_path = os.path.join(models_dir, 'model.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train_model()

