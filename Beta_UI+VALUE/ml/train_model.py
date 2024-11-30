import os
import sys
import torch
from risk_predictor import RiskPredictor

def train_model():
    print("Starting model training...")
    
    # Get the path to the database
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(os.path.dirname(current_dir), 'ai_cdss.db')
    
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return
        
    print(f"Using database at: {db_path}")
    
    # Create predictor with correct database path
    predictor = RiskPredictor()
    predictor.model.db_path = db_path
    
    try:
        # Train model with default parameters
        print("Training model...")
        predictor.model.train(epochs=100, learning_rate=0.01)
        print("Training completed successfully!")
        
        # Verify the model file was created
        model_path = os.path.join(current_dir, 'ml', 'simple_risk_model.pth')
        if os.path.exists(model_path):
            print(f"Model saved successfully to {model_path}")
        else:
            print("Warning: Model file not found after training")
            
        # Test the model on a sample patient
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT patient_id FROM Patients LIMIT 1")
            test_patient = cursor.fetchone()[0]
            conn.close()
            
            # Make a test prediction
            result = predictor.predict(None, test_patient)
            print("\nTest prediction for patient", test_patient)
            print("Risk Score:", result['overall_risk'])
            
        except Exception as e:
            print("Warning: Could not run test prediction:", str(e))
            print("Error details:", str(e))
            
    except Exception as e:
        print("Error during training:", str(e))
        print("Error details:", str(e))
        raise

if __name__ == "__main__":
    train_model()