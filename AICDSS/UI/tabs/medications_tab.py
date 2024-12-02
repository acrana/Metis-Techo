from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                              QComboBox, QPushButton, QGroupBox, QTableWidget,
                              QTableWidgetItem, QMessageBox, QTextEdit)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from datetime import datetime

class MedicationsTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setup_ui()
        
    def setup_ui(self):
        
        layout = QHBoxLayout(self)
        
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Current Medications section
        current_meds_group = QGroupBox("Current Medications")
        current_meds_layout = QVBoxLayout(current_meds_group)
        self.current_meds_table = QTableWidget()
        self.current_meds_table.setColumnCount(2)
        self.current_meds_table.setHorizontalHeaderLabels(["Medication", "Action"])
        self.current_meds_table.horizontalHeader().setStretchLastSection(False)
        self.current_meds_table.horizontalHeader().setMinimumSectionSize(150)
        self.current_meds_table.setColumnWidth(0, 200)
        self.current_meds_table.setColumnWidth(1, 100)
        current_meds_layout.addWidget(self.current_meds_table)
        left_layout.addWidget(current_meds_group)
        
       
        order_group = QGroupBox("New Medication Order")
        order_layout = QVBoxLayout(order_group)
        
        
        med_label = QLabel("Medication:")
        self.med_combo = QComboBox()
        self.med_combo.currentIndexChanged.connect(self.on_medication_select)
        
        
        freq_label = QLabel("Frequency:")
        self.freq_combo = QComboBox()
        self.freq_combo.setEnabled(False)
        self.freq_combo.addItems([
            "Once daily", "Twice daily", "Three times daily", 
            "Four times daily", "Every 6 hours", "Every 8 hours", 
            "Every 12 hours", "As needed"
        ])
        
       
        duration_label = QLabel("Duration:")
        self.duration_combo = QComboBox()
        self.duration_combo.setEnabled(False)
        self.duration_combo.addItems([
            "1 day", "3 days", "5 days", "7 days", "14 days", 
            "30 days", "60 days", "90 days", "Long term"
        ])
        
        
        self.order_button = QPushButton("Place Order")
        self.order_button.setEnabled(False)
        
        
        for widget in [med_label, self.med_combo, freq_label, self.freq_combo,
                      duration_label, self.duration_combo, self.order_button]:
            order_layout.addWidget(widget)
        
        left_layout.addWidget(order_group)
        
        # Right side - Risk Assessment
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Risk Score section
        risk_group = QGroupBox("Risk Assessment")
        risk_layout = QVBoxLayout()
        
        self.risk_score_label = QLabel("No medication selected")
        self.risk_score_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.risk_score_label.setAlignment(Qt.AlignCenter)
        risk_layout.addWidget(self.risk_score_label)
        
        self.risk_details = QTextEdit()
        self.risk_details.setReadOnly(True)
        self.risk_details.setMinimumHeight(100)
        risk_layout.addWidget(self.risk_details)
        
        risk_group.setLayout(risk_layout)
        right_layout.addWidget(risk_group)
        
        
        interactions_group = QGroupBox("Drug Interactions")
        interactions_layout = QVBoxLayout()
        self.interactions_table = QTableWidget()
        self.interactions_table.setColumnCount(2)
        self.interactions_table.setHorizontalHeaderLabels(["Severity", "Details"])
        self.interactions_table.horizontalHeader().setStretchLastSection(True)
        interactions_layout.addWidget(self.interactions_table)
        interactions_group.setLayout(interactions_layout)
        right_layout.addWidget(interactions_group)
        
        
        contra_group = QGroupBox("Contraindications")
        contra_layout = QVBoxLayout()
        self.contraindications_table = QTableWidget()
        self.contraindications_table.setColumnCount(2)
        self.contraindications_table.setHorizontalHeaderLabels(["Risk Level", "Details"])
        self.contraindications_table.horizontalHeader().setStretchLastSection(True)
        contra_layout.addWidget(self.contraindications_table)
        contra_group.setLayout(contra_layout)
        right_layout.addWidget(contra_group)
        
        
        layout.addWidget(left_panel, 1)
        layout.addWidget(right_panel, 2)

    def on_medication_select(self, index):
        
        if index < 0 or not self.main_window.current_patient_id:
            return
            
        medication = self.med_combo.currentText()
        
        # Enable order fields
        self.freq_combo.setEnabled(True)
        self.duration_combo.setEnabled(True)
        self.order_button.setEnabled(True)
        
        try:
            # Get current medications
            current_meds = self.main_window.cdss.get_patient_medications(
                self.main_window.current_patient_id
            )
            
            # Get medication ID for selected medication
            cursor = self.main_window.cdss.conn.cursor()
            cursor.execute("""
                SELECT medication_id, name
                FROM Medications
                WHERE name = ?
            """, (medication,))
            med = cursor.fetchone()
            
            if med:
                # Add selected medication to current medications for interaction check
                current_meds.append((med['medication_id'], med['name']))
                
                # Check interactions
                interactions = self.main_window.cdss.check_medication_interactions(current_meds)
                self.update_interactions_table(interactions)
                
                # Check contraindications
                conditions = self.main_window.cdss.get_patient_conditions(
                    self.main_window.current_patient_id
                )
                contraindications = self.main_window.cdss.check_contraindications(
                    conditions, med['medication_id']
                )
                self.update_contraindications_table(contraindications)
                
                # Get risk assessment
                assessment = self.main_window.risk_predictor.calculate_combined_risk(
                    self.main_window.current_patient_id,
                    medication
                )
                self.update_risk_display(assessment)
                
        except Exception as e:
            print(f"Error in on_medication_select: {e}")

    def update_risk_display(self, assessment):
        
        # Update risk score and level
        score = assessment['total_risk_score']
        level = assessment['risk_level']
        
        # Set color based on risk level
        colors = {
            'HIGH': '#ffcccc',
            'MODERATE': '#fff4cc',
            'LOW': '#ccffcc',
            'MINIMAL': '#ffffff'
        }
        color = colors.get(level, '#ffffff')
        
        self.risk_score_label.setText(f"Risk Level: {level}\nRisk Score: {score:.3f}")
        self.risk_score_label.setStyleSheet(f"background-color: {color}; padding: 10px;")
        
        # Build details text
        details = []
        
        # Add historical ADEs
        ade_details = assessment['components']['historical_ades']
        if ade_details['score'] > 0:
            details.append("Historical ADEs:")
            details.append(f"Score: {ade_details['score']:.3f}")
            for ade in ade_details['details']:
                details.append(f"• {ade['timestamp']}: {ade['description']} (Score: {ade['score']:.3f})")
            details.append("")
        
        # Add vital signs
        vital_details = assessment['components']['vital_trends']
        if vital_details['score'] > 0:
            details.append("Vital Sign Concerns:")
            details.append(f"Score: {vital_details['score']:.3f}")
            for vital, readings in vital_details['trends'].items():
                high_deviations = [r for r in readings if r['deviation'] > 0.3]
                if high_deviations:
                    details.append(f"• {vital}:")
                    for reading in high_deviations[:3]:
                        details.append(
                            f"  {reading['timestamp']}: {reading['value']} "
                            f"(Deviation: {reading['deviation']:.3f})"
                        )
            details.append("")
        
        
        lab_details = assessment['components']['lab_trends']
        if lab_details['score'] > 0:
            details.append("Lab Value Concerns:")
            details.append(f"Score: {lab_details['score']:.3f}")
            for lab, readings in lab_details['trends'].items():
                high_deviations = [r for r in readings if r['deviation'] > 0.3]
                if high_deviations:
                    details.append(f"• {lab}:")
                    for reading in high_deviations[:3]:
                        details.append(
                            f"  {reading['timestamp']}: {reading['value']} "
                            f"(Deviation: {reading['deviation']:.3f})"
                        )
        
        self.risk_details.setText("\n".join(details))

    def update_current_medications(self, medications):
        
        self.current_meds_table.setRowCount(0)
        if not medications:
            return
            
        self.current_meds_table.setRowCount(len(medications))
        for i, med in enumerate(medications):
            
            name_item = QTableWidgetItem(med['name'])
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.current_meds_table.setItem(i, 0, name_item)
            
           
            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            discontinue_btn = QPushButton("Discontinue")
            
            discontinue_btn.clicked.connect(
                lambda checked, m=med: self.main_window.discontinue_medication(m)
            )
            btn_layout.addWidget(discontinue_btn)
            btn_layout.setContentsMargins(4, 2, 4, 2)
            btn_layout.setAlignment(Qt.AlignCenter)
            self.current_meds_table.setCellWidget(i, 1, btn_widget)

    def update_interactions_table(self, interactions):
        
        self.interactions_table.setRowCount(0)
        
        if not interactions:
            return
            
        for interaction in interactions:
            row = self.interactions_table.rowCount()
            self.interactions_table.insertRow(row)
            
            severity_item = QTableWidgetItem(interaction['interaction_type'])
            details_item = QTableWidgetItem(
                f"{interaction['med1_name']} + {interaction['med2_name']}: "
                f"{interaction['description']}"
            )
            
            if interaction['interaction_type'].lower() == 'major':
                severity_item.setBackground(QColor('#ffcccc'))
            elif interaction['interaction_type'].lower() == 'moderate':
                severity_item.setBackground(QColor('#fff4cc'))
            
            self.interactions_table.setItem(row, 0, severity_item)
            self.interactions_table.setItem(row, 1, details_item)

    def update_contraindications_table(self, contraindications):
        
        self.contraindications_table.setRowCount(0)
        
        if not contraindications:
            return
            
        for contra in contraindications:
            row = self.contraindications_table.rowCount()
            self.contraindications_table.insertRow(row)
            
            risk_item = QTableWidgetItem(contra['risk_level'])
            details_item = QTableWidgetItem(
                f"{contra['condition_name']}: {contra['details']}"
            )
            
            if contra['risk_level'].lower() == 'high':
                risk_item.setBackground(QColor('#ffcccc'))
            elif contra['risk_level'].lower() == 'moderate':
                risk_item.setBackground(QColor('#fff4cc'))
            
            self.contraindications_table.setItem(row, 0, risk_item)
            self.contraindications_table.setItem(row, 1, details_item)