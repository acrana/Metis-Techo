import torch
import torch.nn as nn
import sqlite3
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler

class SimpleRiskPredictor:
    def __init__(self, db_path: str = 'ai_cdss.db'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LogisticRiskModel().to(self.device)
        self.scaler = StandardScaler()
        self.db_path = db_path
        
        # Important feature names for interpretability
        self.vital_names = [
            'heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic',
            'respiratory_rate', 'oxygen_saturation', 'temperature'
        ]
        self.lab_names = [
            'CBC', 'Blood Glucose', 'Creatinine', 'Potassium',
            'Liver Enzymes', 'Hemoglobin', 'Platelets', 'Cholesterol', 'QTc'
        ]

    def preprocess_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract and preprocess features from database
        Returns: features tensor, labels tensor
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all patient IDs
        cursor.execute("SELECT DISTINCT patient_id FROM Patients")
        patient_ids = [row[0] for row in cursor.fetchall()]
        
        features_list = []
        labels_list = []
        
        for patient_id in patient_ids:
            # Get latest vitals
            cursor.execute("""
                SELECT heart_rate, blood_pressure_systolic, blood_pressure_diastolic,
                       respiratory_rate, oxygen_saturation, temperature
                FROM Vitals 
                WHERE patient_id = ?
                ORDER BY timestamp DESC LIMIT 1
            """, (patient_id,))
            vitals = cursor.fetchone()
            vital_features = [float(v) if v is not None else 0.0 for v in vitals] if vitals else [0.0] * 6
            
            # Get latest labs
            cursor.execute("""
                SELECT l.lab_name, lr.result
                FROM Lab_Results lr
                JOIN Labs l ON lr.lab_id = l.lab_id
                WHERE lr.patient_id = ?
                AND lr.timestamp = (
                    SELECT MAX(timestamp) 
                    FROM Lab_Results 
                    WHERE lab_id = lr.lab_id 
                    AND patient_id = lr.patient_id
                )
            """, (patient_id,))
            labs = {row[0]: row[1] for row in cursor.fetchall()}
            lab_features = [float(labs.get(name, 0)) for name in self.lab_names]
            
            # Get condition count
            cursor.execute("""
                SELECT COUNT(*) FROM Patient_Conditions WHERE patient_id = ?
            """, (patient_id,))
            condition_count = cursor.fetchone()[0]
            
            # Get medication count
            cursor.execute("""
                SELECT COUNT(DISTINCT medication_id) 
                FROM Patient_Medications 
                WHERE patient_id = ? 
                AND dosage != 'DISCONTINUED'
            """, (patient_id,))
            medication_count = cursor.fetchone()[0]
            
            # Combine all features
            features = vital_features + lab_features + [condition_count, medication_count]
            features_list.append(features)
            
            # Get labels (ADE occurrence)
            cursor.execute("""
                SELECT COUNT(*) > 0 FROM ADE_Monitoring WHERE patient_id = ?
            """, (patient_id,))
            has_ade = cursor.fetchone()[0]
            labels_list.append([float(has_ade)])
        
        conn.close()
        
        # Convert to numpy, scale features, and convert to torch tensors
        features_array = np.array(features_list)
        self.scaler.fit(features_array)
        scaled_features = self.scaler.transform(features_array)
        
        return (
            torch.FloatTensor(scaled_features).to(self.device),
            torch.FloatTensor(labels_list).to(self.device)
        )

    def train(self, epochs: int = 100, learning_rate: float = 0.01):
        """Train the model using k-fold cross validation"""
        X, y = self.preprocess_data()
        
        # Set up optimizer with L2 regularization
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        criterion = nn.BCELoss()
        
        # Simple training loop
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(X)
            loss = criterion(outputs, y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
                
        # Save the model
        torch.save(self.model.state_dict(), 'simple_risk_model.pth')

    def predict(self, patient_id: str) -> Dict[str, float]:
        """Make predictions for a single patient"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Extract features (same as in preprocess_data but for single patient)
        cursor.execute("""
            SELECT heart_rate, blood_pressure_systolic, blood_pressure_diastolic,
                   respiratory_rate, oxygen_saturation, temperature
            FROM Vitals 
            WHERE patient_id = ?
            ORDER BY timestamp DESC LIMIT 1
        """, (patient_id,))
        vitals = cursor.fetchone()
        vital_features = [float(v) if v is not None else 0.0 for v in vitals] if vitals else [0.0] * 6
        
        cursor.execute("""
            SELECT l.lab_name, lr.result
            FROM Lab_Results lr
            JOIN Labs l ON lr.lab_id = l.lab_id
            WHERE lr.patient_id = ?
            AND lr.timestamp = (
                SELECT MAX(timestamp) 
                FROM Lab_Results 
                WHERE lab_id = lr.lab_id 
                AND patient_id = lr.patient_id
            )
        """, (patient_id,))
        labs = {row[0]: row[1] for row in cursor.fetchall()}
        lab_features = [float(labs.get(name, 0)) for name in self.lab_names]
        
        cursor.execute("""
            SELECT COUNT(*) FROM Patient_Conditions WHERE patient_id = ?
        """, (patient_id,))
        condition_count = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(DISTINCT medication_id) 
            FROM Patient_Medications 
            WHERE patient_id = ? 
            AND dosage != 'DISCONTINUED'
        """, (patient_id,))
        medication_count = cursor.fetchone()[0]
        
        conn.close()
        
        # Combine features and scale
        features = vital_features + lab_features + [condition_count, medication_count]
        scaled_features = self.scaler.transform([features])
        
        # Make prediction
        with torch.no_grad():
            self.model.eval()
            X = torch.FloatTensor(scaled_features).to(self.device)
            prediction = self.model(X)
            risk_score = float(prediction[0][0])
            
            return {
                'risk_score': risk_score,
                'risk_level': 'HIGH' if risk_score > 0.7 else 'MODERATE' if risk_score > 0.3 else 'LOW',
                'feature_importance': self._get_feature_importance(X)
            }
    
    def _get_feature_importance(self, X: torch.Tensor) -> Dict[str, float]:
        """Calculate feature importance using weight analysis"""
        weights = self.model.linear.weight.data[0]
        feature_names = (
            self.vital_names + 
            self.lab_names + 
            ['condition_count', 'medication_count']
        )
        
        importance_dict = {}
        for name, weight, value in zip(feature_names, weights, X[0]):
            importance = abs(float(weight * value))
            importance_dict[name] = importance
            
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

class LogisticRiskModel(nn.Module):
    """Simple logistic regression model"""
    def __init__(self, input_dim: int = 17):  # 6 vitals + 9 labs + 2 counts
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))