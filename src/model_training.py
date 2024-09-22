# src/model_training.py

from preprocessing import get_preprocessed_data
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def train_model():
    # Get preprocessed data
    data = get_preprocessed_data()
    
    # Define features and target variable
    X = data.drop(['PatientID', 'Had_ADE'], axis=1)
    y = data['Had_ADE']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Define the model architecture
    model = Sequential([
        Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.1)
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}")
    
    # Save the model
    model.save('models/model.h5')
    print("Model training completed and saved.")

if __name__ == '__main__':
    train_model()
