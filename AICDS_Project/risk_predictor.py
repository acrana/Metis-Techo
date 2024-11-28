import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from ml.patient_risk_model import PatientRiskNet

class RiskPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PatientRiskNet().to(self.device)
        
        # Risk calculation parameters
        self.risk_multipliers = {
            'ade_base': 0.05,
            'ade_max': 0.15,
            'interaction_base': 0.1,
            'interaction_max': 0.2
        }
        
        self.risk_clamps = {
            'min': 0.05,
            'max': 0.75
        }
        
        # Time decay factors (in days)
        self.decay_factors = {
            'ade': 365,
            'vitals': 30,
            'labs': 60
        }
        
        # Features
        self.vital_names = [
            'heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic',
            'respiratory_rate', 'oxygen_saturation', 'temperature'
        ]
        
        self.lab_names = [
            'CBC', 'Blood Glucose', 'Creatinine', 'Potassium',
            'Liver Enzymes', 'Hemoglobin', 'Platelets', 'Cholesterol', 'QTc'
        ]
        
        # Initialize optimizer with L2 regularization
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.01
        )
        
        # Loss function with class weighting
        pos_weight = torch.tensor([2.0, 2.0, 1.5]).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def get_features(self, cursor, patient_id: str, new_med_id: int = None):
        """Extract all features for prediction"""
        # Get vitals
        cursor.execute("""
            SELECT * FROM Vitals 
            WHERE patient_id = ? 
            ORDER BY timestamp DESC LIMIT 1
        """, (patient_id,))
        vitals = cursor.fetchone() or {}
        
        vital_values = []
        for name in self.vital_names:
            try:
                value = float(vitals[name] if vitals else 0)
                vital_values.append(value)
            except (ValueError, TypeError, KeyError):
                vital_values.append(0.0)
        vital_tensor = torch.tensor([vital_values], dtype=torch.float32)
        
        # Get labs
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
        labs = {row['lab_name']: row['result'] for row in cursor.fetchall()}
        
        lab_values = []
        for name in self.lab_names:
            try:
                value = float(labs.get(name, 0))
                lab_values.append(value)
            except (ValueError, TypeError):
                lab_values.append(0.0)
        lab_tensor = torch.tensor([lab_values], dtype=torch.float32)
        
        # Get conditions
        cursor.execute("""
            SELECT condition_id 
            FROM Patient_Conditions 
            WHERE patient_id = ?
        """, (patient_id,))
        conditions = [row['condition_id'] for row in cursor.fetchall()]
        if not conditions:
            conditions = [0]
        condition_tensor = torch.tensor([conditions], dtype=torch.int64)
        
        # Get medications
        cursor.execute("""
            SELECT DISTINCT medication_id 
            FROM Patient_Medications 
            WHERE patient_id = ? 
            AND dosage != 'DISCONTINUED'
        """, (patient_id,))
        medications = [row['medication_id'] for row in cursor.fetchall()]
        if new_med_id:
            medications.append(new_med_id)
        if not medications:
            medications = [0]
        medication_tensor = torch.tensor([medications], dtype=torch.int64)
        
        features = {
            'vitals': vital_tensor.to(self.device),
            'labs': lab_tensor.to(self.device),
            'conditions': condition_tensor.to(self.device),
            'medications': medication_tensor.to(self.device)
        }
        
        if new_med_id:
            features.update(self.get_med_specific_features(cursor, patient_id, new_med_id))
            
        return features

    def get_med_specific_features(self, cursor, patient_id: str, med_id: int) -> Dict:
        """Get medication-specific risk features with proper discontinued med handling"""
        # Get medication type
        cursor.execute("""
            SELECT type FROM Medications WHERE medication_id = ?
        """, (med_id,))
        med_type = cursor.fetchone()['type']
        
        # Get historical ADEs
        cursor.execute("""
            SELECT ade.*, m.type
            FROM ADE_Monitoring ade
            JOIN Medications m ON ade.medication_id = m.medication_id
            WHERE ade.patient_id = ? 
            AND (m.type = ? OR ade.medication_id = ?)
            ORDER BY ade.timestamp DESC
        """, (patient_id, med_type, med_id))
        ades = cursor.fetchall()
        
        ade_score = 0.0
        for ade in ades:
            days_ago = (datetime.now() - datetime.fromisoformat(ade['timestamp'])).days
            time_weight = np.exp(-np.log(2) * days_ago / self.decay_factors['ade'])
            ade_score += time_weight * (1.0 if ade['medication_id'] == med_id else 0.7)
        
        # Modified interaction query to properly handle discontinued medications
        cursor.execute("""
            SELECT
                COUNT(*) as interaction_count,
                COUNT(CASE WHEN mi.interaction_type IN ('Major', 'High') THEN 1 END) as major_count,
                COUNT(CASE WHEN mi.interaction_type = 'Moderate' THEN 1 END) as moderate_count
            FROM (
                SELECT DISTINCT medication_id 
                FROM Patient_Medications 
                WHERE patient_id = ? 
                AND dosage != 'DISCONTINUED'  
                AND medication_id != ?
                AND medication_id IN (
                    SELECT medication_id 
                    FROM Patient_Medications 
                    WHERE patient_id = ?
                    GROUP BY medication_id 
                    HAVING MAX(CASE WHEN dosage = 'DISCONTINUED' THEN 1 ELSE 0 END) = 0
                )
            ) as current_meds
            JOIN Medication_Interactions mi ON 
                (mi.medication_1_id = current_meds.medication_id AND mi.medication_2_id = ?) OR
                (mi.medication_2_id = current_meds.medication_id AND mi.medication_1_id = ?)
        """, (patient_id, med_id, patient_id, med_id, med_id))
        
        interactions = cursor.fetchone()
        
        interaction_score = 0.0
        if interactions['interaction_count'] > 0:
            major_score = (interactions['major_count'] or 0) * 0.25
            moderate_score = (interactions['moderate_count'] or 0) * 0.15
            
            interaction_score = min(major_score + moderate_score, 0.5)
            
            total_meds = interactions['interaction_count']
            if total_meds > 3:
                interaction_score *= (1 + min(total_meds - 3, 3) * 0.1)
        
        med_specific_tensor = torch.tensor([[
            float(ade_score > 0),
            min(ade_score / 2.0, 1.0),
            float(interaction_score > 0),
            interaction_score,
            float(interactions['major_count'] or 0 > 0),
            min(float(interactions['interaction_count']) / 5.0, 1.0)
        ]], dtype=torch.float32).to(self.device)
        
        return {
            'med_specific': med_specific_tensor,
            'risk_factors': {
                'ade_score': ade_score,
                'interaction_score': interaction_score
            }
        }

    def predict(self, cursor, patient_id: str, new_med_id: int = None) -> Dict[str, float]:
        """Generate risk predictions with proper base scoring"""
        self.model.eval()
        with torch.no_grad():
            features = self.get_features(cursor, patient_id, new_med_id)
            
            # If no medications selected, return minimal risk
            if not new_med_id and len(features['medications'][0]) <= 1:  # Only has padding value
                return {
                    'ade_risk': self.risk_clamps['min'],
                    'interaction_risk': self.risk_clamps['min'],
                    'overall_risk': self.risk_clamps['min']
                }
            
            if new_med_id:
                risk_factors = features.pop('risk_factors')
                predictions = self.model(**features)
                
                # Apply adjustments only if real risk factors exist
                ade_multiplier = 1.0
                if risk_factors['ade_score'] > 0:
                    ade_multiplier += min(
                        risk_factors['ade_score'] * self.risk_multipliers['ade_base'],
                        self.risk_multipliers['ade_max']
                    )
                
                int_multiplier = 1.0
                if risk_factors['interaction_score'] > 0:
                    int_multiplier += min(
                        risk_factors['interaction_score'] * self.risk_multipliers['interaction_base'],
                        self.risk_multipliers['interaction_max']
                    )
                
                # Apply multipliers and clamps
                predictions[0, 0] = torch.clamp(
                    predictions[0, 0] * ade_multiplier,
                    self.risk_clamps['min'],
                    self.risk_clamps['max']
                )
                
                predictions[0, 1] = torch.clamp(
                    predictions[0, 1] * int_multiplier,
                    self.risk_clamps['min'],
                    self.risk_clamps['max']
                )
                
                # Calculate overall risk with adjusted weights
                predictions[0, 2] = torch.clamp(
                    (predictions[0, 0] * 0.4 + predictions[0, 1] * 0.6),  # Adjusted weights
                    self.risk_clamps['min'],
                    self.risk_clamps['max']
                )
            else:
                predictions = self.model(**features)
                
            return {
                'ade_risk': float(predictions[0, 0]),
                'interaction_risk': float(predictions[0, 1]),
                'overall_risk': float(predictions[0, 2])
            }

    def train(self, cursor, batch_size: int = 32, epochs: int = 10):
        """Train the model"""
        self.model.train()
        
        # Get all patient IDs
        cursor.execute("SELECT DISTINCT patient_id FROM Patient_Medications")
        patient_ids = [row['patient_id'] for row in cursor.fetchall()]
        
        print(f"Training on {len(patient_ids)} patients")
        
        for epoch in range(epochs):
            total_loss = 0
            batches = 0
            
            np.random.shuffle(patient_ids)
            
            for i in range(0, len(patient_ids), batch_size):
                batch_patients = patient_ids[i:i + batch_size]
                batch_loss = 0
                self.optimizer.zero_grad()
                
                for patient_id in batch_patients:
                    try:
                        features = self.get_features(cursor, patient_id)
                        
                        cursor.execute("""
                            SELECT COUNT(*) as count 
                            FROM ADE_Monitoring 
                            WHERE patient_id = ?
                        """, (patient_id,))
                        has_ade = cursor.fetchone()['count'] > 0
                        
                        cursor.execute("""
                            SELECT COUNT(*) as count 
                            FROM Patient_Medications pm1
                            JOIN Patient_Medications pm2 ON pm1.patient_id = pm2.patient_id
                            JOIN Medication_Interactions mi 
                                ON (mi.medication_1_id = pm1.medication_id 
                                    AND mi.medication_2_id = pm2.medication_id)
                            WHERE pm1.patient_id = ?
                        """, (patient_id,))
                        has_interaction = cursor.fetchone()['count'] > 0
                        
                        labels = torch.tensor([[
                            float(has_ade),
                            float(has_interaction),
                            float(has_ade or has_interaction)
                        ]], dtype=torch.float32).to(self.device)
                        
                        predictions = self.model(**features)
                        loss = self.criterion(predictions, labels)
                        batch_loss += loss
                    
                    except Exception as e:
                        print(f"Error processing patient {patient_id}: {str(e)}")
                        continue
                
                if len(batch_patients) > 0:
                    batch_loss = batch_loss / len(batch_patients)
                    total_loss += batch_loss.item()
                    batches += 1
                    
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
            
            if batches > 0:
                avg_loss = total_loss / batches
                print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

    def explain_prediction(self, cursor, patient_id: str, med_id: int) -> str:
        """Generate an explanation for the prediction"""
        features = self.get_features(cursor, patient_id, med_id)
        risk_factors = features.get('risk_factors', {})

        explanation = []
        if 'ade_score' in risk_factors:
            explanation.append(f"ADE Score: {risk_factors['ade_score']:.2f}")
        if 'interaction_score' in risk_factors:
            explanation.append(f"Interaction Score: {risk_factors['interaction_score']:.2f}")

        return "\n".join(explanation)

    def save(self, path: str):
        """Save model state"""
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        """Load model state"""
        self.model.load_state_dict(torch.load(path))
        self.model.eval()