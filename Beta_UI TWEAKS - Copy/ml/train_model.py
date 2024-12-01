import os
import sys
import sqlite3
import torch
from patient_risk_model import SimpleRiskPredictor

def train_model():
    print("Starting model training...")
    
    # Get directory paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(os.path.dirname(current_dir), 'ai_cdss.db')
    ml_dir = os.path.join(current_dir, 'ml')
    
    # Create ml directory if it doesn't exist
    if not os.path.exists(ml_dir):
        os.makedirs(ml_dir)
        print(f"Created directory: {ml_dir}")
    
    # Define model and scaler paths
    model_path = os.path.join(ml_dir, 'simple_risk_model.pth')
    scaler_path = os.path.join(ml_dir, 'risk_scaler.joblib')
    
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return
        
    print(f"Using database at: {db_path}")
    print(f"Model will be saved to: {model_path}")
    print(f"Scaler will be saved to: {scaler_path}")
    
    predictor = SimpleRiskPredictor(db_path)
    
    try:
        print("\nTraining model...")
        predictor.train(epochs=100, learning_rate=0.01, 
                       model_save_path=model_path,
                       scaler_save_path=scaler_path)
        print("Training completed successfully!")
        
        # Verify both files were created
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            print("\nModel and scaler saved successfully:")
            print(f"Model size: {os.path.getsize(model_path)/1024:.2f} KB")
            print(f"Scaler size: {os.path.getsize(scaler_path)/1024:.2f} KB")
        else:
            missing = []
            if not os.path.exists(model_path):
                missing.append("model")
            if not os.path.exists(scaler_path):
                missing.append("scaler")
            print(f"\nWarning: {', '.join(missing)} file(s) not found after training")
            
        # Test prediction
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT patient_id FROM Patients LIMIT 1")
            test_patient = cursor.fetchone()[0]
            conn.close()
            
            result = predictor.predict(test_patient)
            print("\nTest prediction for patient", test_patient)
            print("Risk Score:", result['risk_score'])
            print("Risk Level:", result['risk_level'])
            
        except Exception as e:
            print("\nWarning: Could not run test prediction:")
            print(f"Error: {str(e)}")
            
    except Exception as e:
        print("\nError during training:")
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()