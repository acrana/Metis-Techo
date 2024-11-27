from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                              QGroupBox, QTextEdit, QTableWidget, 
                              QTableWidgetItem, QSplitter)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

class PatientOverviewTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setup_ui()
        
    def setup_ui(self):
        """Create the patient overview tab"""
        layout = QHBoxLayout(self)
        
        # Create splitter for resizable sections
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Patient information
        info_group = QGroupBox("Patient Information")
        info_layout = QVBoxLayout(info_group)
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        info_layout.addWidget(self.info_text)
        left_layout.addWidget(info_group)
        
        # Conditions
        conditions_group = QGroupBox("Current Conditions")
        conditions_layout = QVBoxLayout(conditions_group)
        self.conditions_table = QTableWidget()
        self.conditions_table.setColumnCount(1)
        self.conditions_table.setHorizontalHeaderLabels(["Condition"])
        self.conditions_table.horizontalHeader().setStretchLastSection(True)
        conditions_layout.addWidget(self.conditions_table)
        left_layout.addWidget(conditions_group)
        
        # Right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Vitals
        vitals_group = QGroupBox("Current Vitals")
        vitals_layout = QVBoxLayout(vitals_group)
        self.vitals_widgets = {}
        
        vital_types = [
            ('heart_rate', 'Heart Rate', 'bpm'),
            ('blood_pressure_systolic', 'Blood Pressure', 'mmHg'),
            ('respiratory_rate', 'Respiratory Rate', '/min'),
            ('oxygen_saturation', 'O2 Saturation', '%'),
            ('temperature', 'Temperature', '°C')
        ]
        
        for vital_id, name, unit in vital_types:
            vital_frame = QWidget()
            vital_layout = QHBoxLayout(vital_frame)
            
            label = QLabel(f"{name}:")
            label.setMinimumWidth(150)
            value = QLabel("--")
            value.setMinimumWidth(80)
            unit_label = QLabel(unit)
            trend = QLabel("")
            trend.setMinimumWidth(20)
            
            vital_layout.addWidget(label)
            vital_layout.addWidget(value)
            vital_layout.addWidget(unit_label)
            vital_layout.addWidget(trend)
            vital_layout.addStretch()
            
            self.vitals_widgets[vital_id] = {
                'value': value,
                'trend': trend
            }
            
            vitals_layout.addWidget(vital_frame)
        
        right_layout.addWidget(vitals_group)
        
        # Current medications
        meds_group = QGroupBox("Current Medications")
        meds_layout = QVBoxLayout(meds_group)
        self.current_meds_table = QTableWidget()
        self.current_meds_table.setColumnCount(1)
        self.current_meds_table.setHorizontalHeaderLabels(["Medication"])
        self.current_meds_table.horizontalHeader().setStretchLastSection(True)
        meds_layout.addWidget(self.current_meds_table)
        right_layout.addWidget(meds_group)
        
        # Add panels to splitter
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        
        layout.addWidget(splitter)

    def update_patient_info(self, patient_info):
        """Update patient information display"""
        if patient_info:
            info_text = (
                f"Name: {patient_info['name']}\n"
                f"Age: {patient_info['age']}\n"
                f"Gender: {patient_info['gender']}\n"
                f"Admission Date: {patient_info['admission_date']}\n"
                f"Discharge Date: {patient_info['discharge_date'] or 'Not discharged'}\n"
                f"Reason for Admission: {patient_info['reason_for_admission']}\n"
                f"ICU Patient: {'Yes' if patient_info['icu'] else 'No'}\n"
                f"Frequent Vital Monitoring: {'Yes' if patient_info['q1h_vitals'] else 'No'}\n"
                f"Frequent Medication Changes: {'Yes' if patient_info['frequent_med_changes'] else 'No'}"
            )
            self.info_text.setText(info_text)

    def update_conditions(self, conditions):
        """Update conditions table"""
        self.conditions_table.setRowCount(0)
        if not conditions:
            return
            
        self.conditions_table.setRowCount(len(conditions))
        for i, (condition_name, _) in enumerate(conditions):
            item = QTableWidgetItem(condition_name)
            self.conditions_table.setItem(i, 0, item)

    def update_vitals(self, vitals, vital_ranges):
        """Update vital signs display"""
        if not vitals:
            return
            
        vital_pairs = [
            ("heart_rate", vitals['heart_rate']),
            ("blood_pressure_systolic", 
             f"{vitals['blood_pressure_systolic']}/{vitals['blood_pressure_diastolic']}"),
            ("respiratory_rate", vitals['respiratory_rate']),
            ("oxygen_saturation", vitals['oxygen_saturation']),
            ("temperature", vitals['temperature'])
        ]
        
        for vital_id, value in vital_pairs:
            if vital_id in self.vitals_widgets:
                widgets = self.vitals_widgets[vital_id]
                widgets['value'].setText(str(value))
                
                if vital_id in vital_ranges:
                    ranges = vital_ranges[vital_id]
                    try:
                        if isinstance(value, str) and '/' in value:
                            sys_val = float(value.split('/')[0])
                            value_for_check = sys_val
                        else:
                            value_for_check = float(value)
                            
                        if value_for_check <= ranges['critical_min'] or \
                           value_for_check >= ranges['critical_max']:
                            widgets['value'].setStyleSheet("background-color: #ffcccc;")
                        elif value_for_check < ranges['min_normal'] or \
                             value_for_check > ranges['max_normal']:
                            widgets['value'].setStyleSheet("background-color: #fff4cc;")
                        else:
                            widgets['value'].setStyleSheet("background-color: #ccffcc;")
                    except (ValueError, TypeError):
                        widgets['value'].setStyleSheet("")

    def update_medications(self, medications):
        """Update current medications table"""
        self.current_meds_table.setRowCount(0)
        if not medications:
            return
            
        self.current_meds_table.setRowCount(len(medications))
        for i, med_details in enumerate(medications):
            self.current_meds_table.setItem(i, 0, QTableWidgetItem(med_details['name']))
