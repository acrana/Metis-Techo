import sys
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLabel, QComboBox, QTabWidget, QLineEdit)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from cdss_functions import CDSSSystem
from UI.tabs.overview_tab import PatientOverviewTab
from UI.tabs.medications_tab import MedicationsTab
from UI.tabs.analytics_tab import PatientAnalyticsTab
from UI.tabs.ml_risk_tab import MLRiskTab
from UI.main_window.patient_handlers import PatientHandlersMixin
from UI.handlers.medication_handler import MedicationHandlersMixin
from ml.ml_risk_predictor import EnhancedRiskPredictor
from risk_predictor import RiskPredictor

class CDSSUI(QMainWindow, PatientHandlersMixin, MedicationHandlersMixin):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clinical Decision Support System")
        self.setMinimumSize(1400, 800)
        
        # Initialize CDSS system and risk predictors
        self.cdss = CDSSSystem()
        self.risk_predictor = EnhancedRiskPredictor(self.cdss)
        self.ml_predictor = RiskPredictor()
        
        # Load trained model if exists
        model_path = os.path.join(os.path.dirname(__file__), 'ml', 'trained_models', 'patient_risk_model.pth')
        if os.path.exists(model_path):
            self.ml_predictor.load(model_path)
        
        # Track current selections
        self.current_patient_id = None
        self.current_medication = None
        self.medications_data = []
        self.vital_ranges = {}
        self.lab_ranges = {}
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Create UI components
        self.create_header_section(main_layout)
        self.create_content_section(main_layout)
        
        # Load initial data
        self.load_initial_data()
        
        # Setup auto-refresh timer
        self.setup_refresh_timer()

    def create_header_section(self, parent_layout):
        """Create the header section with patient selection"""
        header = QWidget()
        header_layout = QHBoxLayout(header)
        
        # Patient selection
        select_label = QLabel("Select Patient:")
        select_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.patient_combo = QComboBox()
        self.patient_combo.setMinimumWidth(300)
        self.patient_combo.currentIndexChanged.connect(self.on_patient_select)
        
        # Search
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        self.search_input = QLineEdit()
        self.search_input.textChanged.connect(self.filter_patients)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_input)
        
        # Patient info display
        self.patient_info = QLabel()
        self.patient_info.setFont(QFont("Arial", 11))
        
        header_layout.addWidget(select_label)
        header_layout.addWidget(self.patient_combo)
        header_layout.addLayout(search_layout)
        header_layout.addWidget(self.patient_info)
        header_layout.addStretch()
        
        parent_layout.addWidget(header)

    def create_content_section(self, parent_layout):
        """Create the main content area with tabs"""
        self.tab_widget = QTabWidget()
        
        # Create main tabs
        self.create_overview_tab()
        self.create_medications_tab()
        self.create_analytics_tab()
        self.create_ml_risk_tab()
        
        parent_layout.addWidget(self.tab_widget)

    def create_overview_tab(self):
        """Create the patient overview tab"""
        self.overview_tab = PatientOverviewTab(self)
        self.tab_widget.addTab(self.overview_tab, "Patient Overview")

    def create_medications_tab(self):
        """Create the medications tab"""
        self.medications_tab = MedicationsTab(self)
        self.medications_tab.order_button.clicked.connect(self.place_order)
        self.tab_widget.addTab(self.medications_tab, "Medications")

    def create_analytics_tab(self):
        """Create the analytics tab"""
        self.analytics_tab = PatientAnalyticsTab(self)
        self.tab_widget.addTab(self.analytics_tab, "Analytics")

    def create_ml_risk_tab(self):
        """Create the ML risk analysis tab"""
        self.ml_risk_tab = MLRiskTab(self)
        self.tab_widget.addTab(self.ml_risk_tab, "ML Risk Analysis")

    def update_ml_risk_analysis(self):
        """Update ML risk analysis when patient or medication changes"""
        if not self.current_patient_id:
            return
            
        # Get medication ID if one is selected
        medication_id = None
        if self.current_medication:
            cursor = self.cdss.conn.cursor()
            cursor.execute("""
                SELECT medication_id 
                FROM Medications 
                WHERE name = ?
            """, (self.current_medication,))
            result = cursor.fetchone()
            if result:
                medication_id = result['medication_id']
        
        self.ml_risk_tab.update_display(self.current_patient_id, medication_id)

    def load_initial_data(self):
        """Load all initial data"""
        self.load_patients()
        self.load_medications()
        self.load_ranges()

    def setup_refresh_timer(self):
        """Setup timer for refreshing vitals"""
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_vitals)
        self.refresh_timer.start(60000)  # Refresh every minute

    def load_ranges(self):
        """Load vital and lab ranges"""
        cursor = self.cdss.conn.cursor()
        
        # Load vital ranges
        cursor.execute("SELECT * FROM Vital_Ranges")
        self.vital_ranges = {
            row['vital_name']: dict(row) 
            for row in cursor.fetchall()
        }
        
        # Load lab ranges
        cursor.execute("SELECT * FROM Lab_Ranges")
        self.lab_ranges = {
            row['lab_name']: dict(row) 
            for row in cursor.fetchall()
        }

    def on_patient_select(self, index):
        """Handle patient selection"""
        if index < 0:
            return
            
        selection = self.patient_combo.currentText()
        self.current_patient_id = selection.split('(')[1].rstrip(')')
        
        # Update all patient information
        self.update_patient_info()
        self.update_conditions()
        self.update_current_medications()
        self.update_vitals()
        self.update_ml_risk_analysis()
        
        # Enable medication selection in medications tab
        if hasattr(self, 'medications_tab'):
            self.medications_tab.med_combo.setEnabled(True)
        if hasattr(self, 'analytics_tab'):
            self.analytics_tab.update_analytics(self.current_patient_id)

    def on_medication_select(self, index):
        """Handle medication selection"""
        if index < 0:
            return
            
        medication = self.medications_tab.med_combo.currentText()
        self.current_medication = medication
        
        # Enable other fields
        self.medications_tab.freq_combo.setEnabled(True)
        self.medications_tab.duration_combo.setEnabled(True)
        self.medications_tab.order_button.setEnabled(True)
        
        # Update safety information
        self.update_safety_check()
        self.update_ml_risk_analysis()

    def closeEvent(self, event):
        """Handle application closure"""
        self.refresh_timer.stop()
        self.cdss.close()
        super().closeEvent(event)