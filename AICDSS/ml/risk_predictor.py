import os
import torch
from typing import Dict, Any, List
try:

    from .patient_risk_model import SimpleRiskPredictor
except ImportError:

    from patient_risk_model import SimpleRiskPredictor

class RiskPredictor:
    def __init__(self):
        self.model = SimpleRiskPredictor()
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, 'ml', 'simple_risk_model.pth')
        scaler_path = os.path.join(base_path, 'ml', 'risk_scaler.joblib')
        
        try:
            print(f"Loading model from: {model_path}")
            print(f"Loading scaler from: {scaler_path}")
            self.model.load(model_path, scaler_path)
            print("Loaded model and scaler successfully.")
        except FileNotFoundError:
            print(f"Model or scaler files not found. Please train the model first.")
        except Exception as e:
            print(f"Error loading model: {e}")

    def get_structured_features(self, raw_features: Dict[str, float], X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert flat feature dictionary to structured format expected by UI"""
        
        vital_names = self.model.vital_names
        lab_names = self.model.lab_names
        
        vital_values = []
        lab_values = []
        other_values = []
        
        vital_section = X[0][:len(vital_names)]
        lab_section = X[0][len(vital_names):len(vital_names) + len(lab_names)]
        other_section = X[0][len(vital_names) + len(lab_names):]
        
        return {
            'vitals': vital_section.unsqueeze(0),  # Add batch dimension
            'labs': lab_section.unsqueeze(0),
            'other': other_section.unsqueeze(0),
            'vital_names': vital_names,
            'lab_names': lab_names,
            'importance': raw_features  # Keep original importance scores
        }

    def get_features(self, cursor, patient_id: str, new_med_id: int = None):
        """Get feature importance"""
        try:
           
            result = self.model.predict(patient_id)

   
            filtered_importance = {
                name: value
                for name, value in result['feature_importance'].items()
                if name != 'medication_count'  # Exclude medication_count
            }

          
            return self.get_structured_features(
                filtered_importance,
                self.model.last_input_tensor
            )
        except Exception as e:
            print(f"Feature extraction error: {str(e)}")
            return {
                'vitals': torch.zeros(1, len(self.model.vital_names)),
                'labs': torch.zeros(1, len(self.model.lab_names)),
                'other': torch.zeros(1, 2),  # condition_count and medication_count
                'vital_names': self.model.vital_names,
                'lab_names': self.model.lab_names,
                'importance': {}
            }

    def get_risk_analysis(self, cursor, patient_id: str, medication_id: int, features: Dict) -> List[str]:
        """Generate detailed risk analysis text"""
        risk_text = []

        try:
            # Get medication info and its type
            cursor.execute("""
                SELECT name, type FROM Medications WHERE medication_id = ?
            """, (medication_id,))
            med = cursor.fetchone()
            if med:
                risk_text.append(f"Analysis for: {med['name']} ({med['type']})\n")
                med_type = med['type']

           
            if 'importance' in features:
                top_features = sorted(features['importance'].items(), key=lambda x: x[1], reverse=True)[:5]
                risk_text.append("Key Risk Factors:")
                for feature, importance in top_features:
                    if importance > 0.1:  # Only show significant factors
                        # Add medication-specific context
                        context = ""
                        if feature in ['heart_rate', 'blood_pressure_systolic'] and med_type in ['Beta-blocker', 'Antiarrhythmic']:
                            context = " (relevant for " + med_type + ")"
                        elif feature in ['creatinine', 'potassium'] and med_type in ['ACE Inhibitor', 'Diuretic']:
                            context = " (relevant for " + med_type + ")"
                        risk_text.append(f"• {feature}: {importance:.3f}{context}")
                risk_text.append("")

           
            cursor.execute("""
                SELECT DISTINCT c.condition_name, 
                            CASE WHEN cc.medication_id IS NOT NULL THEN 'High Risk'
                                WHEN c.condition_id IN (
                                    SELECT condition_id 
                                    FROM Condition_Contraindications 
                                    WHERE medication_id = ?
                                ) THEN 'Contraindicated'
                                ELSE 'Present'
                            END as relevance
                FROM Patient_Conditions pc
                JOIN Conditions c ON pc.condition_id = c.condition_id
                LEFT JOIN Condition_Contraindications cc ON 
                    c.condition_id = cc.condition_id AND 
                    cc.medication_id = ?
                WHERE pc.patient_id = ?
                ORDER BY relevance DESC
            """, (medication_id, medication_id, patient_id))
        
            conditions = cursor.fetchall()
            if conditions:
                risk_text.append("Relevant Conditions:")
                for condition in conditions:
                    relevance_marker = ""
                    if condition['relevance'] == 'Contraindicated':
                        relevance_marker = " ⚠️ CONTRAINDICATED"
                    elif condition['relevance'] == 'High Risk':
                        relevance_marker = " ⚠️ HIGH RISK"
                    risk_text.append(f"• {condition['condition_name']}{relevance_marker}")
                risk_text.append("")

            
            cursor.execute("""
                WITH MedicationClass AS (
                    SELECT type FROM Medications WHERE medication_id = ?
                )
                SELECT am.description, am.timestamp, 
                    m.name as med_name, m.type as med_type,
                    CASE WHEN m.medication_id = ? THEN 'Same Medication'
                            WHEN m.type = (SELECT type FROM MedicationClass) THEN 'Same Class'
                            ELSE 'Other'
                    END as relevance
                FROM ADE_Monitoring am
                JOIN Medications m ON am.medication_id = m.medication_id
                WHERE am.patient_id = ? AND 
                    (m.medication_id = ? OR m.type = (SELECT type FROM MedicationClass))
                ORDER BY 
                    CASE WHEN m.medication_id = ? THEN 1
                        WHEN m.type = (SELECT type FROM MedicationClass) THEN 2
                        ELSE 3
                    END,
                    am.timestamp DESC
                LIMIT 5
            """, (medication_id, medication_id, patient_id, medication_id, medication_id))
        
            ades = cursor.fetchall()
            if ades:
                risk_text.append("Recent Adverse Events:")
                for ade in ades:
                    relevance_marker = ""
                    if ade['relevance'] == 'Same Medication':
                        relevance_marker = " ⚠️ SAME MEDICATION"
                    elif ade['relevance'] == 'Same Class':
                        relevance_marker = " ⚠️ SAME CLASS"
                    risk_text.append(
                        f"• {ade['timestamp'].split()[0]}: {ade['description']} "
                        f"({ade['med_name']}){relevance_marker}"
                    )

            return risk_text

        except Exception as e:
            print(f"Error generating risk analysis: {e}")
            return ["Error generating risk analysis"]

    def predict(self, cursor, patient_id: str, new_med_id: int = None) -> Dict[str, Any]:
        """Make predictions for a patient"""
        try:
            result = self.model.predict(patient_id, new_med_id)
            risk_score = float(result['risk_score'])
        
          
            features = self.get_structured_features(
                result['feature_importance'],
                self.model.last_input_tensor
            )
        
            
            risk_analysis = []
            if new_med_id:
                risk_analysis = self.get_risk_analysis(cursor, patient_id, new_med_id, features)
        
            return {
                'ade_risk': risk_score,
                'interaction_risk': risk_score * 0.8,
                'overall_risk': risk_score,
                'vital_risk': risk_score * 0.7,
                'risk_level': result['risk_level'],
                'vital_warnings': [],
                'features': features,
                'risk_analysis': risk_analysis
            }
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise