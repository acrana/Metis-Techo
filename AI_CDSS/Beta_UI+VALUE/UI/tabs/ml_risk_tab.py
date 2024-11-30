from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                              QGroupBox, QTextEdit, QTableWidget, 
                              QTableWidgetItem, QPushButton, QComboBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor

class MLRiskTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setup_ui()
        
    def setup_ui(self):
        """Create the ML risk analysis tab"""
        layout = QHBoxLayout(self)
        
        # Left panel - Neural Network Predictions
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Medication Selection
        med_group = QGroupBox("Medication Selection")
        med_layout = QVBoxLayout(med_group)
        
        # Add medication selector
        self.med_label = QLabel("Select Medication to Analyze:")
        self.med_combo = QComboBox()
        self.med_combo.currentIndexChanged.connect(self.on_medication_select)
        med_layout.addWidget(self.med_label)
        med_layout.addWidget(self.med_combo)
        med_group.setLayout(med_layout)
        left_layout.addWidget(med_group)
        
        # ML Risk Score Overview
        risk_group = QGroupBox("Neural Network Risk Analysis")
        risk_layout = QVBoxLayout(risk_group)
        
        # Create labels for risk scores
        self.ade_risk_label = QLabel("ADE Risk: --")
        self.interaction_risk_label = QLabel("Interaction Risk: --")
        self.overall_risk_label = QLabel("Overall Risk: --")
        
        for label in [self.ade_risk_label, self.interaction_risk_label, self.overall_risk_label]:
            label.setFont(QFont("Arial", 12))
            risk_layout.addWidget(label)
            
        risk_group.setLayout(risk_layout)
        left_layout.addWidget(risk_group)
        
        # Feature Importance section
        feature_group = QGroupBox("Feature Analysis")
        feature_layout = QVBoxLayout(feature_group)
        
        # Add explanation label
        feature_explanation = QLabel("Analysis of input features and their impact on risk scores")
        feature_explanation.setWordWrap(True)
        feature_layout.addWidget(feature_explanation)
        
        self.feature_table = QTableWidget()
        self.feature_table.setColumnCount(3)
        self.feature_table.setHorizontalHeaderLabels(["Feature", "Value", "Impact"])
        self.feature_table.horizontalHeader().setStretchLastSection(True)
        feature_layout.addWidget(self.feature_table)
        feature_group.setLayout(feature_layout)
        left_layout.addWidget(feature_group)
        
        # Right panel - Risk Factors
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Risk Factor Analysis
        factors_group = QGroupBox("Risk Factor Analysis")
        factors_layout = QVBoxLayout(factors_group)
        
        # Add explanation label
        factors_explanation = QLabel("Detailed breakdown of identified risk factors")
        factors_explanation.setWordWrap(True)
        factors_layout.addWidget(factors_explanation)
        
        self.risk_details = QTextEdit()
        self.risk_details.setReadOnly(True)
        factors_layout.addWidget(self.risk_details)
        factors_group.setLayout(factors_layout)
        right_layout.addWidget(factors_group)
        
        # Model Details
        model_group = QGroupBox("Model Information")
        model_layout = QVBoxLayout(model_group)
        
        self.model_info = QTextEdit()
        self.model_info.setReadOnly(True)
        self.model_info.setText(
            "Neural Network Architecture:\n"
            "- Input Features: Patient vitals, labs, conditions, medications\n"
            "- Hidden Layers: 64 -> 32 neurons\n"
            "- Dropout: 0.3, 0.4\n"
            "- Output: 3 risk scores (ADE, Interaction, Overall)\n\n"
            "Training Parameters:\n"
            "- Optimizer: Adam (lr=0.001)\n"
            "- Loss: Binary Cross Entropy with Logits\n"
            "- Class Weighting: [2.0, 2.0, 1.5]\n"
            "- L2 Regularization: 0.01"
        )
        model_layout.addWidget(self.model_info)
        model_group.setLayout(model_layout)
        right_layout.addWidget(model_group)
        
        left_panel.setLayout(left_layout)
        right_panel.setLayout(right_layout)
        
        # Add panels to main layout
        layout.addWidget(left_panel)
        layout.addWidget(right_panel)
        
        # Load medications
        self.load_medications()

    def load_medications(self):
        """Load available medications into combo box"""
        cursor = self.main_window.cdss.conn.cursor()
        cursor.execute("""
            SELECT medication_id, name 
            FROM Medications 
            ORDER BY name
        """)
        self.med_combo.clear()
        self.med_combo.addItem("Select a medication...", None)
        for med in cursor.fetchall():
            self.med_combo.addItem(med['name'], med['medication_id'])

    def on_medication_select(self, index):
        """Handle medication selection"""
        if index <= 0:  # No selection or first item
            return
            
        medication_id = self.med_combo.currentData()
        if medication_id and self.main_window.current_patient_id:
            self.update_display(self.main_window.current_patient_id, medication_id)

    def update_for_new_patient(self):
        """Update when a new patient is selected"""
        if self.main_window.current_patient_id:
            self.med_combo.setEnabled(True)
            # Clear previous analysis
            self.update_display(self.main_window.current_patient_id)
        else:
            self.med_combo.setEnabled(False)
            self.med_combo.setCurrentIndex(0)

    def update_display(self, patient_id: str = None, medication_id: int = None):
        """Update the display with new predictions"""
        if not patient_id:
            return
            
        try:
            # Get predictions from the PyTorch model
            predictions = self.main_window.ml_predictor.predict(
                self.main_window.cdss.conn.cursor(),
                patient_id,
                medication_id
            )
            
            # Update risk labels
            self.ade_risk_label.setText(f"ADE Risk: {predictions['ade_risk']:.1%}")
            self.interaction_risk_label.setText(f"Interaction Risk: {predictions['interaction_risk']:.1%}")
            self.overall_risk_label.setText(f"Overall Risk: {predictions['overall_risk']:.1%}")
            
            # Color code based on risk level
            for label, risk in [
                (self.ade_risk_label, predictions['ade_risk']),
                (self.interaction_risk_label, predictions['interaction_risk']),
                (self.overall_risk_label, predictions['overall_risk'])
            ]:
                if risk >= 0.7:
                    label.setStyleSheet("color: red; font-weight: bold;")
                elif risk >= 0.4:
                    label.setStyleSheet("color: orange; font-weight: bold;")
                else:
                    label.setStyleSheet("color: green;")
            
            # Get features and analysis
            features = self.main_window.ml_predictor.get_features(
                self.main_window.cdss.conn.cursor(),
                patient_id,
                medication_id
            )
            
            # Update feature table
            self.feature_table.setRowCount(0)
            
            # Add vital signs
            vital_tensor = features['vitals']
            for i, name in enumerate(self.main_window.ml_predictor.vital_names):
                row = self.feature_table.rowCount()
                self.feature_table.insertRow(row)
                self.feature_table.setItem(row, 0, QTableWidgetItem(f"Vital: {name}"))
                self.feature_table.setItem(row, 1, QTableWidgetItem(f"{vital_tensor[0][i].item():.1f}"))
                impact = abs(vital_tensor[0][i].item() - 1.0)
                self.feature_table.setItem(row, 2, QTableWidgetItem(f"{impact:.3f}"))
            
            # Add lab values
            lab_tensor = features['labs']
            for i, name in enumerate(self.main_window.ml_predictor.lab_names):
                row = self.feature_table.rowCount()
                self.feature_table.insertRow(row)
                self.feature_table.setItem(row, 0, QTableWidgetItem(f"Lab: {name}"))
                self.feature_table.setItem(row, 1, QTableWidgetItem(f"{lab_tensor[0][i].item():.1f}"))
                impact = abs(lab_tensor[0][i].item() - 1.0)
                self.feature_table.setItem(row, 2, QTableWidgetItem(f"{impact:.3f}"))
            
            # Add medication-specific features if available
            if medication_id and 'risk_factors' in features:
                factors = features['risk_factors']
                risk_text = []
                
                cursor = self.main_window.cdss.conn.cursor()
                cursor.execute("""
                    SELECT name, type FROM Medications WHERE medication_id = ?
                """, (medication_id,))
                med = cursor.fetchone()
                if med:
                    risk_text.append(f"Analysis for: {med['name']} ({med['type']})\n")
                
                if 'ade_score' in factors:
                    score = factors['ade_score']
                    risk_text.append(f"Historical ADE Risk Score: {score:.3f}")
                    self.add_feature_row("Historical ADEs", score)
                
                if 'interaction_score' in factors:
                    score = factors['interaction_score']
                    risk_text.append(f"Drug Interaction Score: {score:.3f}")
                    self.add_feature_row("Drug Interactions", score)
                
                # Add historical ADEs
                cursor.execute("""
                    SELECT description, timestamp 
                    FROM ADE_Monitoring
                    WHERE patient_id = ? 
                    ORDER BY timestamp DESC LIMIT 3
                """, (patient_id,))
                ades = cursor.fetchall()
                if ades:
                    risk_text.append("\nRelevant Historical ADEs:")
                    for ade in ades:
                        risk_text.append(f"• {ade['timestamp']}: {ade['description']}")
                
                self.risk_details.setText("\n".join(risk_text))
            else:
                self.risk_details.setText("Select a medication to see detailed risk analysis")
        
        except Exception as e:
            print(f"Error updating ML risk display: {str(e)}")

    def add_feature_row(self, name: str, value: float):
        """Helper to add a row to the feature table"""
        row = self.feature_table.rowCount()
        self.feature_table.insertRow(row)
        self.feature_table.setItem(row, 0, QTableWidgetItem(name))
        self.feature_table.setItem(row, 1, QTableWidgetItem(f"{value:.3f}"))
        self.feature_table.setItem(row, 2, QTableWidgetItem(f"{value:.3f}"))