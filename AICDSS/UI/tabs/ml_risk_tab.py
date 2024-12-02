from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                              QGroupBox, QTextEdit, QTableWidget, 
                              QTableWidgetItem, QPushButton, QComboBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor
from typing import Dict, Any

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

        left_panel.setLayout(left_layout)  # Set the layout for left panel
        
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
        
        # Summary section
        summary_group = QGroupBox("Risk Assessment Summary")
        summary_layout = QVBoxLayout(summary_group)

        # Current Status Overview
        status_label = QLabel("Current Status")
        status_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(100)

        # Trend Analysis
        trend_label = QLabel("30-Day Trend Analysis")
        trend_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.trend_text = QTextEdit()
        self.trend_text.setReadOnly(True)
        self.trend_text.setMaximumHeight(100)

        # Recommendations
        recommendations_label = QLabel("Clinical Decision Support")
        recommendations_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.recommendations_text = QTextEdit()
        self.recommendations_text.setReadOnly(True)
        self.recommendations_text.setMaximumHeight(100)

        # Add all widgets to summary layout
        for widget in [status_label, self.status_text, 
                    trend_label, self.trend_text,
                    recommendations_label, self.recommendations_text]:
            summary_layout.addWidget(widget)

        summary_group.setLayout(summary_layout)
        right_layout.addWidget(summary_group)

        right_panel.setLayout(right_layout)  # Set the layout for right panel

        # Add both panels to main layout
        layout.addWidget(left_panel)
        layout.addWidget(right_panel)

        # Load medications
        self.load_medications()

    def update_summary(self, predictions: Dict[str, Any]):
        """Update the summary section with meaningful insights"""
        if not predictions:
            return
        
        # Update Current Status
        status_text = [
            "Risk Assessment Status:",
            f"• Overall Risk Level: {predictions['risk_level']}",
            f"• ADE Risk: {predictions['ade_risk']:.1%}",
            f"• Interaction Risk: {predictions['interaction_risk']:.1%}\n"
        ]
        self.status_text.setText("\n".join(status_text))
    
        # Update Trend Analysis
        if 'features' in predictions:
            features = predictions['features']
            concerning_vitals = []
            concerning_labs = []
        
            # Check vitals
            if 'vitals' in features and 'vital_names' in features:
                vitals = features['vitals'][0]
                for i, name in enumerate(features['vital_names']):
                    if abs(float(vitals[i])) > 1.5:  # More than 1.5 std dev from mean
                        concerning_vitals.append(name)
        
            # Check labs
            if 'labs' in features and 'lab_names' in features:
                labs = features['labs'][0]
                for i, name in enumerate(features['lab_names']):
                    if abs(float(labs[i])) > 1.5:  # More than 1.5 std dev from mean
                        concerning_labs.append(name)
        
            trend_text = ["📈 Key Trends:"]
            if concerning_vitals:
                trend_text.append("• Concerning Vital Signs: " + ", ".join(concerning_vitals))
            if concerning_labs:
                trend_text.append("• Concerning Lab Values: " + ", ".join(concerning_labs))
            if not (concerning_vitals or concerning_labs):
                trend_text.append("• No significant deviations in monitored parameters")
        
            self.trend_text.setText("\n".join(trend_text))
    
        # Update Recommendations
        recommendations = ["💡 Suggested Actions:"]
    
        risk_level = predictions['risk_level']
        if risk_level == 'HIGH':
            recommendations.extend([
                "• Consider alternative medication options",
                "• Implement enhanced monitoring protocol",
                "• Review drug interactions and contraindications",
                "• Schedule frequent reassessment"
            ])
        elif risk_level == 'MODERATE':
            recommendations.extend([
                "• Monitor key parameters more frequently",
                "• Consider dose adjustment if appropriate",
                "• Schedule regular follow-up"
            ])
        else:
            recommendations.extend([
                "• Proceed with standard monitoring",
                "• Continue regular assessment schedule"
            ])
    
        self.recommendations_text.setText("\n".join(recommendations))

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
            try:
                print(f"Medication selected: {self.med_combo.currentText()} for patient {self.main_window.current_patient_id}")
                # Force a full update of the display with current patient and selected medication
                self.update_display(self.main_window.current_patient_id, medication_id)
            except Exception as e:
                print(f"Error in medication selection: {str(e)}")
                self.risk_details.setText("Error updating analysis")

    def update_for_new_patient(self):
        """Update when a new patient is selected"""
        if self.main_window.current_patient_id:
            self.med_combo.setEnabled(True)
            # Get currently selected medication if any
            medication_id = self.med_combo.currentData() if self.med_combo.currentIndex() > 0 else None
            # Update display with current patient and medication
            self.update_display(self.main_window.current_patient_id, medication_id)
        else:
            self.med_combo.setEnabled(False)
            self.med_combo.setCurrentIndex(0)

    def update_display(self, patient_id: str = None, medication_id: int = None):
        if not patient_id:
            return
    
        print(f"Updating display for patient {patient_id} with medication {medication_id}")
        
        try:
            # Clear previous predictions if no medication selected
            if medication_id is None:
                self.ade_risk_label.setText("ADE Risk: --")
                self.interaction_risk_label.setText("Interaction Risk: --")
                self.overall_risk_label.setText("Overall Risk: --")
                self.feature_table.setRowCount(0)
                self.risk_details.setText("Select a medication to see detailed risk analysis")
                return

        # Get fresh predictions from the PyTorch model
            predictions = self.main_window.ml_predictor.predict(
                self.main_window.cdss.conn.cursor(),
                patient_id,
                medication_id
            )

            print(f"Got predictions: {predictions['risk_level']}")

            self.update_summary(predictions)
    
        # Update risk labels with new predictions
            current_ade = f"ADE Risk: {predictions['ade_risk']:.1%}"
            print(f"Setting ADE risk to: {current_ade}")
            self.ade_risk_label.setText(current_ade)
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
    
        # Clear and update feature table
            self.feature_table.setRowCount(0)
            features = predictions.get('features', {})
    
        # Add vital signs
            vital_tensor = features.get('vitals')
            vital_names = features.get('vital_names', [])
            if vital_tensor is not None and vital_names:
                for i, name in enumerate(vital_names):
                    if i < vital_tensor.size(1):
                        row = self.feature_table.rowCount()
                        self.feature_table.insertRow(row)
                        self.feature_table.setItem(row, 0, QTableWidgetItem(f"Vital: {name}"))
                        self.feature_table.setItem(row, 1, QTableWidgetItem(f"{vital_tensor[0][i].item():.1f}"))
                        importance = features.get('importance', {}).get(name, 0.0)
                        self.feature_table.setItem(row, 2, QTableWidgetItem(f"{importance:.3f}"))
    
        # Add lab values
            lab_tensor = features.get('labs')
            lab_names = features.get('lab_names', [])
            if lab_tensor is not None and lab_names:
                for i, name in enumerate(lab_names):
                    if i < lab_tensor.size(1):
                        row = self.feature_table.rowCount()
                        self.feature_table.insertRow(row)
                        self.feature_table.setItem(row, 0, QTableWidgetItem(f"Lab: {name}"))
                        self.feature_table.setItem(row, 1, QTableWidgetItem(f"{lab_tensor[0][i].item():.1f}"))
                        importance = features.get('importance', {}).get(name, 0.0)
                        self.feature_table.setItem(row, 2, QTableWidgetItem(f"{importance:.3f}"))
    
        # Update risk details text
            if medication_id and 'risk_analysis' in predictions:
                self.risk_details.setText("\n".join(predictions['risk_analysis']))
            else:
                self.risk_details.setText("Select a medication to see detailed risk analysis")
        
        except Exception as e:
            print(f"Error updating ML risk display: {str(e)}")
            self.risk_details.setText("Error updating risk analysis")

    def add_feature_row(self, name: str, value: float):
        """Helper to add a row to the feature table"""
        row = self.feature_table.rowCount()
        self.feature_table.insertRow(row)
        self.feature_table.setItem(row, 0, QTableWidgetItem(name))
        self.feature_table.setItem(row, 1, QTableWidgetItem(f"{value:.3f}"))
        self.feature_table.setItem(row, 2, QTableWidgetItem(f"{value:.3f}"))