import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from ml.patient_risk_model import PatientRiskNet
from collections import defaultdict

class RiskPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PatientRiskNet().to(self.device)
        
        # Risk calculation parameters with improved scaling
        self.risk_multipliers = {
            'ade_base': 0.15,
            'ade_max': 0.35,
            'interaction_base': 0.20,
            'interaction_max': 0.40
        }
        
        self.risk_clamps = {
            'min': 0.05,
            'max': 0.85
        }
        
        # Time decay factors (in days) with more granular control
        self.decay_factors = {
            'ade': 180,       # 6 months for ADE history
            'vitals': 30,     # 1 month for vitals
            'labs': 60,       # 2 months for labs
            'med_history': 90 # 3 months for medication history
        }
        
        # High-risk medications configuration
        self.high_risk_meds = {
            9: {'name': 'Digoxin', 'ade_mult': 1.5, 'int_mult': 1.3},
            12: {'name': 'Amiodarone', 'ade_mult': 1.4, 'int_mult': 1.6},
            2: {'name': 'Warfarin', 'ade_mult': 1.6, 'int_mult': 1.4}
        }
        
        # Known dangerous drug combinations
        self.high_risk_pairs = {
            frozenset([2, 1]): 0.7,   # Warfarin + Aspirin
            frozenset([12, 9]): 0.8,  # Amiodarone + Digoxin
            frozenset([2, 12]): 0.6   # Warfarin + Amiodarone
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

    def calculate_time_weight(self, days_ago: float, decay_type: str) -> float:
        """Calculate time-based weight for historical events"""
        decay_factor = self.decay_factors.get(decay_type, 180)
        return math.exp(-math.log(2) * days_ago / decay_factor)

    def get_medication_history_weight(self, cursor, patient_id: str) -> float:
        """Calculate weighted impact of medication history"""
        query = """
            SELECT timestamp, dosage
            FROM Patient_Medications
            WHERE patient_id = ?
            ORDER BY timestamp DESC
        """
        cursor.execute(query, (patient_id,))
        
        total_weight = 0.0
        med_changes = defaultdict(list)
        
        for record in cursor.fetchall():
            days_ago = (datetime.now() - datetime.fromisoformat(record['timestamp'])).days
            weight = self.calculate_time_weight(days_ago, 'med_history')
            
            # Add extra weight for discontinuations
            if record['dosage'] == 'DISCONTINUED':
                weight *= 1.2
            
            med_changes[record['medication_id']].append({
                'days_ago': days_ago,
                'weight': weight,
                'dosage': record['dosage']
            })
            
            total_weight += weight
        
        # Add additional weight for frequent changes
        for med_id, changes in med_changes.items():
            if len(changes) > 2:
                recent_changes = sum(1 for c in changes if c['days_ago'] <= 90)
                total_weight += recent_changes * 0.1
        
        return min(total_weight, 2.0)

    def get_med_risk_modifiers(self, med_id: int) -> Dict[str, float]:
        """Get risk multipliers for specific medications"""
        if med_id in self.high_risk_meds:
            return {
                'ade_multiplier': self.high_risk_meds[med_id]['ade_mult'],
                'interaction_multiplier': self.high_risk_meds[med_id]['int_mult']
            }
        return {'ade_multiplier': 1.0, 'interaction_multiplier': 1.0}

    def calculate_interaction_risk(self, cursor, patient_id: str, new_med_id: int = None) -> float:
        """Calculate interaction risk based on medication combinations"""
        query = """
            SELECT DISTINCT medication_id 
            FROM Patient_Medications 
            WHERE patient_id = ? 
            AND dosage != 'DISCONTINUED'
        """
        cursor.execute(query, (patient_id,))
        current_meds = [row['medication_id'] for row in cursor.fetchall()]
        
        if new_med_id:
            current_meds.append(new_med_id)
        
        risk_score = 0.0
        interaction_details = []
        
        # Check known high-risk combinations
        for i, med1 in enumerate(current_meds):
            for med2 in current_meds[i+1:]:
                pair = frozenset([med1, med2])
                if pair in self.high_risk_pairs:
                    risk_score += self.high_risk_pairs[pair]
                    interaction_details.append((med1, med2, self.high_risk_pairs[pair]))
        
        # Add complexity factor for multiple medications
        if len(current_meds) > 3:
            complexity_factor = min((len(current_meds) - 3) * 0.1, 0.3)
            risk_score *= (1 + complexity_factor)
        
        return min(risk_score, 1.0), interaction_details

    def predict(self, cursor, patient_id: str, new_med_id: int = None) -> Dict[str, float]:
        """Generate risk predictions with improved calculation"""
        self.model.eval()
        with torch.no_grad():
            base_features = self.get_features(cursor, patient_id, new_med_id)
            
            # Calculate medication history impact
            med_history_weight = self.get_medication_history_weight(cursor, patient_id)
            
            # Get interaction risk
            interaction_risk, interaction_details = self.calculate_interaction_risk(
                cursor, patient_id, new_med_id
            )
            
            # Calculate base risks
            base_predictions = self.model(**base_features)
            
            # Apply medication-specific modifiers
            if new_med_id:
                risk_modifiers = self.get_med_risk_modifiers(new_med_id)
                
                # Adjust ADE risk
                ade_risk = float(base_predictions[0, 0]) * risk_modifiers['ade_multiplier']
                ade_risk = min(ade_risk * (1 + med_history_weight * 0.2), self.risk_clamps['max'])
                
                # Adjust interaction risk
                base_interaction = float(base_predictions[0, 1])
                modified_interaction = base_interaction * risk_modifiers['interaction_multiplier']
                final_interaction = max(modified_interaction, interaction_risk)
                
                # Calculate overall risk
                overall_risk = (ade_risk * 0.4 + final_interaction * 0.6)
            else:
                ade_risk = float(base_predictions[0, 0])
                final_interaction = interaction_risk
                overall_risk = float(base_predictions[0, 2])
            
            # Apply risk clamps
            predictions = {
                'ade_risk': max(min(ade_risk, self.risk_clamps['max']), self.risk_clamps['min']),
                'interaction_risk': max(min(final_interaction, self.risk_clamps['max']), self.risk_clamps['min']),
                'overall_risk': max(min(overall_risk, self.risk_clamps['max']), self.risk_clamps['min'])
            }
            
            return predictions

    def explain_prediction(self, cursor, patient_id: str, med_id: int) -> str:
        """Generate detailed explanation for prediction"""
        base_features = self.get_features(cursor, patient_id, med_id)
        risk_modifiers = self.get_med_risk_modifiers(med_id)
        med_history_weight = self.get_medication_history_weight(cursor, patient_id)
        interaction_risk, interaction_details = self.calculate_interaction_risk(cursor, patient_id, med_id)
        
        explanation = []
        
        # Medication-specific risks
        if med_id in self.high_risk_meds:
            med_info = self.high_risk_meds[med_id]
            explanation.append(f"High-risk medication: {med_info['name']}")
            explanation.append(f"ADE Risk Multiplier: {med_info['ade_mult']:.1f}x")
            explanation.append(f"Interaction Risk Multiplier: {med_info['int_mult']:.1f}x")
        
        # Medication history impact
        if med_history_weight > 1.0:
            explanation.append(f"Medication History Impact: +{(med_history_weight-1.0)*100:.0f}%")
        
        # Interaction details
        if interaction_details:
            explanation.append("\nPotential Interactions:")
            for med1, med2, risk in interaction_details:
                explanation.append(f"- {self.get_med_name(cursor, med1)} with {self.get_med_name(cursor, med2)}: {risk:.0%} risk")
        
        return "\n".join(explanation)

    def get_med_name(self, cursor, med_id: int) -> str:
        """Helper function to get medication name"""
        cursor.execute("SELECT name FROM Medications WHERE medication_id = ?", (med_id,))
        result = cursor.fetchone()
        return result['name'] if result else f"Medication {med_id}"
