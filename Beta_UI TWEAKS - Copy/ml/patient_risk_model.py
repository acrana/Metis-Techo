import torch
import torch.nn as nn
import sqlite3
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
import joblib

class SimpleRiskPredictor:
    def __init__(self, db_path: str = 'ai_cdss.db'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LogisticRiskModel().to(self.device)
        self.scaler = StandardScaler()
        self.db_path = db_path
        
        self.vital_names = [
            'heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic',
            'respiratory_rate', 'oxygen_saturation', 'temperature'
        ]
        self.lab_names = [
            'CBC', 'Blood Glucose', 'Creatinine', 'Potassium',
            'Liver Enzymes', 'Hemoglobin', 'Platelets', 'Cholesterol', 'QTc'
        ]

    def preprocess_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
            
            # Define clinical relevance mappings
        clinical_relevance = {
            'Beta-blocker': {
                'vitals': {'heart_rate': 2.5, 'blood_pressure_systolic': 2.0},
                'labs': {
                    'potassium': 0.8,     # Important but not primary
                    'creatinine': 0.5,    # Monitor but less critical
                    'qtc': 1.0            # Relevant for bradycardia risk
                }
            },
            'Anticoagulant': {
                'labs': {
                    'platelets': 2.5,     # Critical for bleeding risk
                    'hemoglobin': 2.5,    # Critical for bleeding risk
                    'inr': 2.5            # Critical for monitoring
                },
                'vitals': {
                    'blood_pressure_systolic': 1.0  # Monitor for bleeding risk
                }
            },
            'ACE Inhibitor': {
                'vitals': {'blood_pressure_systolic': 2.0},
                'labs': {
                    'potassium': 2.0,     # Critical for hyperkalemia
                    'creatinine': 2.0,    # Critical for kidney function
                    'blood_glucose': 0.3   # Less relevant
                }
            },
            'Antiarrhythmic': {
                'vitals': {'heart_rate': 2.0},
                'labs': {
                    'qtc': 2.5,           # Critical for arrhythmia risk
                    'potassium': 2.0,     # Critical for arrhythmia risk
                    'magnesium': 1.5,     # Important for rhythm
                    'creatinine': 0.8     # Monitor but secondary
                }
            },
            'Diuretic': {
                'labs': {
                    'potassium': 2.5,     # Critical for monitoring
                    'creatinine': 2.0,    # Critical for kidney function
                    'blood_glucose': 0.5   # Monitor but less critical
                },
                'vitals': {
                    'blood_pressure_systolic': 1.5,
                    'heart_rate': 0.8     # Monitor for dehydration
                }
            },
            'Statin': {
                'labs': {
                    'liver_enzymes': 2.0,  # Critical for hepatotoxicity
                    'creatinine': 0.8,     # Monitor for rhabdo
                    'creatine_kinase': 1.5 # Important for muscle toxicity
                }
            },
            'Antipsychotic': {
                'labs': {
                    'qtc': 2.0,           # Critical for cardiac risk
                    'blood_glucose': 1.5,  # Important for metabolic syndrome
                    'liver_enzymes': 1.0   # Monitor but secondary
                },
                'vitals': {
                    'heart_rate': 1.0,     # Monitor for tachycardia
                    'blood_pressure_systolic': 1.0  # Monitor for orthostasis
                }
            }
        }

        # Get all patient IDs
        cursor.execute("SELECT DISTINCT patient_id FROM Patients")
        patient_ids = [row[0] for row in cursor.fetchall()]
        
        features_list = []
        labels_list = []
        
        for patient_id in patient_ids:
            # Get latest vitals and medication type
            cursor.execute("""
                SELECT heart_rate, blood_pressure_systolic, blood_pressure_diastolic,
                       respiratory_rate, oxygen_saturation, temperature,
                       (SELECT type FROM Medications m 
                        JOIN Patient_Medications pm ON m.medication_id = pm.medication_id 
                        WHERE pm.patient_id = v.patient_id 
                        AND pm.dosage != 'DISCONTINUED' 
                        ORDER BY pm.timestamp DESC LIMIT 1) as med_type
                FROM Vitals v
                WHERE v.patient_id = ?
                ORDER BY v.timestamp DESC LIMIT 1
            """, (patient_id,))
            vitals = cursor.fetchone()
            
            if vitals:
                med_type = vitals[6]  # Get medication type
                vital_weights = clinical_relevance.get(med_type, {}).get('vitals', {}) if med_type else {}
                
                vital_features = [
                    float(vitals[0]) * vital_weights.get('heart_rate', 0.5) if vitals[0] is not None else 0.0,
                    float(vitals[1]) * vital_weights.get('blood_pressure_systolic', 0.5) if vitals[1] is not None else 0.0,
                    0.0,  # blood_pressure_diastolic
                    0.0,  # respiratory_rate
                    0.0,  # oxygen_saturation
                    0.0   # temperature
                ]
            else:
                vital_features = [0.0] * 6
            
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
            
            labs = {}
            lab_weights = clinical_relevance.get(med_type, {}).get('labs', {}) if med_type else {}
            
            for row in cursor.fetchall():
                try:
                    value = float(row[1]) if row[1] is not None else 0.0
                    weight = lab_weights.get(row[0].lower(), 0.5)  # Default weight 0.5 for non-critical labs
                    labs[row[0]] = value * weight
                except (ValueError, TypeError):
                    labs[row[0]] = 0.0
            
            lab_features = [labs.get(name, 0.0) for name in self.lab_names]
            
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
            
            # Combine features
            features = vital_features + lab_features + [condition_count, medication_count]
            features_list.append(features)
            
            # Get labels
            cursor.execute("""
                SELECT COUNT(*) > 0 FROM ADE_Monitoring WHERE patient_id = ?
            """, (patient_id,))
            has_ade = cursor.fetchone()[0]
            labels_list.append([float(has_ade)])
        
        conn.close()
        
        # Convert to numpy arrays
        features_array = np.array(features_list)
        scaled_features = self.scaler.fit_transform(features_array)
        
        return (
            torch.FloatTensor(scaled_features).to(self.device),
            torch.FloatTensor(labels_list).to(self.device)
        )

    def save(self, model_path: str, scaler_path: str):
        """Save both model and scaler states"""
        torch.save(self.model.state_dict(), model_path)
        joblib.dump(self.scaler, scaler_path)
        
    def load(self, model_path: str, scaler_path: str):
        """Load both model and scaler states"""
        self.model.load_state_dict(torch.load(model_path))
        self.scaler = joblib.load(scaler_path)
        
    def train(self, epochs: int = 100, learning_rate: float = 0.01,
          model_save_path: str = 'simple_risk_model.pth',
          scaler_save_path: str = 'risk_scaler.joblib'):
        """Train the model and save both model and scaler"""
        X, y = self.preprocess_data()
        
        optimizer = torch.optim.Adam(self.model.parameters(), 
                               lr=0.003,  # Reduced from 0.01
                               weight_decay=0.04)  # Increased from 0.01
        pos_weight = torch.tensor([0.7])  # Reduce weight of positive predictions
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
                
        # Save the model
        print("\nSaving model and scaler...")
        self.save(model_save_path, scaler_save_path)
        print("Save completed!")

    def predict(self, patient_id: str, medication_id: str = None) -> Dict[str, float]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get base features as before
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
        
        # Get condition count
        cursor.execute("""
            SELECT COUNT(*) FROM Patient_Conditions WHERE patient_id = ?
        """, (patient_id,))
        condition_count = cursor.fetchone()[0]
        
        # Get medication count and interaction potential
        med_risk_factor = 1.0
        if medication_id:
            # Check for medication interactions
            cursor.execute("""
                SELECT COUNT(*) 
                FROM Patient_Medications pm
                JOIN Medication_Interactions mi ON 
                    (pm.medication_id = mi.medication_1_id AND mi.medication_2_id = ?) OR
                    (pm.medication_id = mi.medication_2_id AND mi.medication_1_id = ?)
                WHERE pm.patient_id = ? 
                AND pm.dosage != 'DISCONTINUED'
                AND mi.interaction_type IN ('Major', 'High')
            """, (medication_id, medication_id, patient_id))
            interaction_count = cursor.fetchone()[0]
            
            # Check for contraindications
            cursor.execute("""
                SELECT COUNT(*)
                FROM Patient_Conditions pc
                JOIN Condition_Contraindications cc ON pc.condition_id = cc.condition_id
                WHERE pc.patient_id = ? AND cc.medication_id = ? AND cc.risk_level = 'High'
            """, (patient_id, medication_id))
            contraindication_count = cursor.fetchone()[0]
            
            # Adjust risk factor based on interactions and contraindications
            med_risk_factor += (interaction_count * 0.15 + contraindication_count * 0.25)
        
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
        
        # Make prediction with medication risk adjustment
        with torch.no_grad():
            self.model.eval()
            X = torch.FloatTensor(scaled_features).to(self.device)
            self.last_input_tensor = X
            prediction = self.model(X)
            risk_score = float(prediction[0][0]) * med_risk_factor
            
            # Ensure risk score stays within valid range
            risk_score = min(max(risk_score, 0.0), 1.0)
        
            return {
                'risk_score': risk_score,
                'risk_level': 'HIGH' if risk_score > 0.7 else 'MODERATE' if risk_score > 0.3 else 'LOW',
                'feature_importance': self._get_feature_importance(X)
            }
    
    def _get_feature_importance(self, X: torch.Tensor) -> Dict[str, float]:
        try:
            weights = self.model.linear.weight.data[0]
            feature_names = (
                self.vital_names + 
                self.lab_names + 
                ['condition_count', 'medication_count']
            )
            
            importance_dict = {}
            
            # Add minimum threshold for importance
            MIN_IMPORTANCE = 0.08  # Only show features with > 5% importance
            
            for name, weight, value in zip(feature_names, weights, X[0]):
                if name == 'medication_count':
                    continue
                    
                weight_val = float(weight.item())
                value_val = float(value.item())
                importance = abs(weight_val * value_val)
                
                # Only add if above threshold
                if importance > MIN_IMPORTANCE:
                    importance_dict[name] = float(importance)

            return dict(sorted(
                importance_dict.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
        except Exception as e:
            print(f"Error calculating feature importance: {str(e)}")
            return {}


class LogisticRiskModel(nn.Module):
    """Simple logistic regression model"""
    def __init__(self, input_dim: int = 17):  # 6 vitals + 9 labs + 2 counts
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))