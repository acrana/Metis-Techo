# risk_predictor.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from typing import Dict
from patient_risk_model import PatientRiskNet

class RiskPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PatientRiskNet().to(self.device)
        self.med_specific_layer = nn.Linear(1, 16)
        
        # Initialize vital and lab names
        self.vital_names = [
            'heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic',
            'respiratory_rate', 'oxygen_saturation', 'temperature'
        ]
        
        # Updated to match your Lab_Ranges table
        self.lab_names = [
            'CBC', 'Blood Glucose', 'Creatinine', 'Potassium',
            'Liver Enzymes', 'Hemoglobin', 'Platelets', 'Cholesterol', 'QTc'
        ]

    def get_features(self, cursor, patient_id: str, new_med_id: int = None):
        features = super().get_features(cursor, patient_id)
        cursor.execute("""
            SELECT * FROM Vitals 
            WHERE patient_id = ? 
            ORDER BY timestamp DESC LIMIT 1
        """, (patient_id,))
        vitals = cursor.fetchone() or {}
        
        # Convert vitals to tensor
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
        
        # Convert labs to tensor
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
            cursor.execute("""
                SELECT type FROM Medications WHERE medication_id = ?
            """, (new_med_id,)) 
            med_type = cursor.fetchone()['type']

            cursor.execute("""
                SELECT COUNT(*) as count
                FROM ADE_Monitoring ade
                JOIN Medications m ON ade.medication_id = m.medication_id
                WHERE ade.patient_id = ? 
                AND (ade.medication_id = ? OR m.type = ?)    
            """, (patient_id, new_med_id, med_type))
            specific_ade_count = cursor.fetchone()['count']  

            features['med_specific'] = torch.tensor([[
                float(specific_ade_count > 0),  # Had ADE with this med/class
        ]], dtype=torch.float32).to(self.device) 
            return features


    def predict(self, cursor, patient_id: str, new_med_id: int = None) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            features = self.get_features(cursor, patient_id, new_med_id)
            predictions = self.model(**features)

            if new_med_id is not None:
                if 'med_specific' in features and features['med_specific'][0][0] == 0:
                    predictions = predictions * 0.7

            
            return {
                'ade_risk': float(predictions[0, 0]),
                'interaction_risk': float(predictions[0, 1]),
                'overall_risk': float(predictions[0, 2])
            }

    def train(self, cursor, batch_size: int = 32, epochs: int = 10):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters())
        criterion = nn.BCELoss()
        
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
                optimizer.zero_grad()
                
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
                        loss = criterion(predictions, labels)
                        batch_loss += loss
                    
                    except Exception as e:
                        print(f"Error processing patient {patient_id}: {str(e)}")
                        continue
                
                if len(batch_patients) > 0:
                    batch_loss = batch_loss / len(batch_patients)
                    total_loss += batch_loss.item()
                    batches += 1
                    
                    batch_loss.backward()
                    optimizer.step()
            
            if batches > 0:
                avg_loss = total_loss / batches
                print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def explain_prediction(self, cursor, patient_id: str, new_med_id: int = None):
        """Explain what factors contribute to the risk prediction"""
        try:
            # Get patient info
            cursor.execute("""
                SELECT name, age FROM Patients WHERE patient_id = ?
            """, (patient_id,))
            patient = cursor.fetchone()
            
            # Get current medications
            cursor.execute("""
                SELECT m.name, m.medication_id 
                FROM Patient_Medications pm
                JOIN Medications m ON pm.medication_id = m.medication_id
                WHERE pm.patient_id = ? AND pm.dosage != 'DISCONTINUED'
            """, (patient_id,))
            current_meds = cursor.fetchall()
            
            # Get most recent vitals
            cursor.execute("""
                SELECT * FROM Vitals
                WHERE patient_id = ?
                ORDER BY timestamp DESC LIMIT 1
            """, (patient_id,))
            vitals = cursor.fetchone()
            vitals_dict = dict(zip([d[0] for d in cursor.description], vitals)) if vitals else {}
            
            # Get vital ranges from your constant ranges
            vital_ranges = {
                'heart_rate': {'min_normal': 60, 'max_normal': 100},
                'blood_pressure_systolic': {'min_normal': 90, 'max_normal': 140},
                'blood_pressure_diastolic': {'min_normal': 60, 'max_normal': 90},
                'respiratory_rate': {'min_normal': 12, 'max_normal': 20},
                'oxygen_saturation': {'min_normal': 95, 'max_normal': 100},
                'temperature': {'min_normal': 36.5, 'max_normal': 37.5}
            }
            
            # Get recent labs
            cursor.execute("""
                SELECT 
                    lr.result,
                    l.lab_name,
                    lr.timestamp,
                    lr.lab_id,
                    ranges.min_normal,
                    ranges.max_normal,
                    ranges.critical_min,
                    ranges.critical_max
                FROM Lab_Results lr
                JOIN Labs l ON lr.lab_id = l.lab_id
                LEFT JOIN Lab_Ranges ranges ON l.lab_name = ranges.lab_name
                WHERE lr.patient_id = ?
                AND lr.timestamp = (
                    SELECT MAX(timestamp) 
                    FROM Lab_Results 
                    WHERE lab_id = lr.lab_id 
                    AND patient_id = lr.patient_id
                )
                ORDER BY l.lab_name
            """, (patient_id,))
            labs = cursor.fetchall()
            
            # Get drug interactions
            interactions = []
            for med1 in current_meds:
                for med2 in current_meds:
                    if med1['medication_id'] < med2['medication_id']:
                        cursor.execute("""
                            SELECT * FROM Medication_Interactions
                            WHERE (medication_1_id = ? AND medication_2_id = ?)
                            OR (medication_1_id = ? AND medication_2_id = ?)
                        """, (med1['medication_id'], med2['medication_id'],
                             med2['medication_id'], med1['medication_id']))
                        interaction = cursor.fetchone()
                        if interaction:
                            interactions.append({
                                'med1': med1['name'],
                                'med2': med2['name'],
                                'type': interaction['interaction_type'],
                                'description': interaction['description']
                            })
            
            # Get previous ADEs
            cursor.execute("""
            SELECT m.name, ade.description
            FROM ADE_Monitoring ade
            JOIN Medications m ON ade.medication_id = m.medication_id
            WHERE ade.patient_id = ?
            ORDER BY ade.timestamp DESC
            """, (patient_id,))
            ades = cursor.fetchall()

        

            
            # Get predictions
            predictions = self.predict(cursor, patient_id, new_med_id)
            
            # Build explanation
            explanation = []
            explanation.append(f"Risk Analysis for {patient['name']} (ID: {patient_id})")
            explanation.append(f"Age: {patient['age']}")
            
            explanation.append("\nCurrent Medications:")
            for med in current_meds:
                explanation.append(f"- {med['name']}")
            
            if vitals:
                explanation.append("\nVital Signs:")
                vital_concerns = []
                vital_mapping = {
                    'heart_rate': ('Heart Rate', 'bpm'),
                    'blood_pressure_systolic': ('Systolic BP', 'mmHg'),
                    'blood_pressure_diastolic': ('Diastolic BP', 'mmHg'),
                    'respiratory_rate': ('Respiratory Rate', '/min'),
                    'oxygen_saturation': ('O2 Saturation', '%'),
                    'temperature': ('Temperature', '°C')
                }
                
                for db_name, (display_name, unit) in vital_mapping.items():
                    if db_name in vitals_dict:
                        try:
                            value = float(vitals_dict[db_name])
                            ranges = vital_ranges[db_name]
                            min_val = float(ranges['min_normal'])
                            max_val = float(ranges['max_normal'])
                            
                            status = "NORMAL"
                            if value < min_val:
                                status = "LOW"
                                vital_concerns.append(f"{display_name} is low")
                            elif value > max_val:
                                status = "HIGH"
                                vital_concerns.append(f"{display_name} is high")
                            explanation.append(f"- {display_name}: {value} {unit} ({status})")
                        except (ValueError, TypeError):
                            continue
                
                if vital_concerns:
                    explanation.append("\nVital Sign Concerns:")
                    for concern in vital_concerns:
                        explanation.append(f"- {concern}")
            
            if labs:
                explanation.append("\nRecent Lab Values:")
                lab_concerns = []
                for lab in labs:
                    lab_dict = dict(zip([d[0] for d in cursor.description], lab))
                    try:
                        value = float(lab[0])
                        lab_name = lab[1]
                        min_val = float(lab[4] or 0)
                        max_val = float(lab[5] or 999999)
                        crit_min = float(lab[6] or 0)
                        crit_max = float(lab[7] or 999999)
                        
                        if value <= crit_min:
                            status = "CRITICALLY LOW"
                            lab_concerns.append(f"{lab_name} is critically low ({value})")
                        elif value >= crit_max:
                            status = "CRITICALLY HIGH"
                            lab_concerns.append(f"{lab_name} is critically high ({value})")
                        elif value < min_val:
                            status = "LOW"
                            lab_concerns.append(f"{lab_name} is low ({value})")
                        elif value > max_val:
                            status = "HIGH"
                            lab_concerns.append(f"{lab_name} is high ({value})")
                        else:
                            status = "NORMAL"
            
                        explanation.append(f"- {lab_name}: {value} ({status})")
                    except (ValueError, TypeError, IndexError) as e:
                        print(f"Error processing lab value: {str(e)}")
                        continue
                
                if lab_concerns:
                    explanation.append("\nLab Value Concerns:")
                    for concern in lab_concerns:
                        explanation.append(f"- {concern}")
            
            if interactions:
                explanation.append("\nDrug Interactions:")
                for interaction in interactions:
                    explanation.append(f"- {interaction['med1']} + {interaction['med2']}")
                    explanation.append(f"  Type: {interaction['type']}")
                    explanation.append(f"  Details: {interaction['description']}")
            
            if ades:
                explanation.append("\nPrevious Adverse Drug Events:")
                for ade in ades:
                    ade_dict = dict(zip([d[0] for d in cursor.description], ade))
                    explanation.append(f"- {ade[0]}: {ade[1]}")
            
            explanation.append("\nRisk Assessment:")
            explanation.append(f"ADE Risk: {predictions['ade_risk']:.1%}")
            if predictions['ade_risk'] > 0.7:
                explanation.append("  ⚠️ HIGH risk of adverse drug events")
            elif predictions['ade_risk'] > 0.3:
                explanation.append("  ⚠️ MODERATE risk of adverse drug events")
            
            explanation.append(f"Interaction Risk: {predictions['interaction_risk']:.1%}")
            if predictions['interaction_risk'] > 0.7:
                explanation.append("  ⚠️ HIGH risk of drug interactions")
            elif predictions['interaction_risk'] > 0.3:
                explanation.append("  ⚠️ MODERATE risk of drug interactions")
            
            explanation.append(f"Overall Risk: {predictions['overall_risk']:.1%}")
            if predictions['overall_risk'] > 0.7:
                explanation.append("  ⚠️ HIGH overall risk - careful monitoring recommended")
            elif predictions['overall_risk'] > 0.3:
                explanation.append("  ⚠️ MODERATE overall risk - monitoring recommended")
            
            return "\n".join(explanation)
            
        except Exception as e:
            import traceback
            return f"Error generating explanation: {str(e)}\n{traceback.format_exc()}"