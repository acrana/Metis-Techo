import sys
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLabel, QComboBox, QTabWidget, QLineEdit, QPushButton)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QIcon
from cdss_functions import CDSSSystem
from UI.tabs.overview_tab import PatientOverviewTab
from UI.tabs.medications_tab import MedicationsTab
from UI.tabs.analytics_tab import PatientAnalyticsTab
from UI.tabs.ml_risk_tab import MLRiskTab
from UI.main_window.patient_handlers import PatientHandlersMixin
from UI.handlers.medication_handler import MedicationHandlersMixin
from ml.ml_risk_predictor import EnhancedRiskPredictor
from ml.risk_predictor import RiskPredictor
from PySide6.QtGui import QPixmap
from PySide6.QtGui import QFontDatabase




class CDSSUI(QMainWindow, PatientHandlersMixin, MedicationHandlersMixin):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("METIS")
        QApplication.setFont(QFont("Segoe UI", 10))
        
        
        icon_path = os.path.join(os.path.dirname(__file__), "metis_icon.png")  # Replace with your icon file name
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            print(f"Warning: Icon file not found at {icon_path}")

        self.setMinimumSize(1400, 800)
        
       
        self.cdss = CDSSSystem()
        self.risk_predictor = EnhancedRiskPredictor(self.cdss)
        self.ml_predictor = RiskPredictor()
        
        
        model_path = os.path.join(os.path.dirname(__file__), 'ml', 'trained_models', 'patient_risk_model.pth')
        if os.path.exists(model_path):
            self.ml_predictor.load(model_path)
        
       
        self.current_patient_id = None
        self.current_medication = None
        self.medications_data = []
        self.vital_ranges = {}
        self.lab_ranges = {}

        
        self.is_dark_mode = False
        
      
        self.dark_mode_stylesheet = """
            QWidget {
                background-color: #2c2c2e;
                color: #e5e5e7;
            }
            QPushButton {
                background-color: #3a3a3c;
                color: #f2f2f7;
                border: 1px solid #48484a;
                border-radius: 5px;
                padding: 7px;
            }
            QPushButton:hover {
                background-color: #48484a;
            }
            QLineEdit, QComboBox {
                background-color: #3a3a3c;
                color: #f2f2f7;
                border: 1px solid #636366;
                border-radius: 3px;
            }
            QLabel {
                color: #d1d1d6;
            }
            QLabel.highlight {
                background-color: #ffffff;
                color: #000000;
                padding: 3px;
            }
            QTabWidget::pane {
                background: #3a3a3c;
                border: 1px solid #48484a;
            }
            QTabBar::tab {
                background: #3a3a3c;
                color: #f2f2f7;
                border: 1px solid #48484a;
                padding: 5px;
                border-radius: 3px;
            }
            QTabBar::tab:selected {
                background: #48484a;
            }
            QTableWidget {
                background-color: #2c2c2e;
                color: #f2f2f7;
                alternate-background-color: #353535;
                border: 1px solid #444444;
            }
            QGroupBox {
                border: 1px solid #444444;
                color: #e5e5e7;
                margin-top: 10px;
                border-radius: 5px;
            }
        """
        self.light_mode_stylesheet = """
            QWidget {
                background-color: #f7f7f7;
                color: #333333;
            }
            QPushButton {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #d0d0d0;
                border-radius: 5px;
                padding: 7px;
            }
            QPushButton:hover {
                background-color: #e6e6e6;
            }
            QLineEdit, QComboBox {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
            QLabel {
                color: #333333;
            }
            QLabel.highlight {
                background-color: #ffffff;
                color: #000000;
                padding: 3px;
            }
            QTabWidget::pane {
                background: #ffffff;
                border: 1px solid #cccccc;
            }
            QTabBar::tab {
                background: #f0f0f0;
                color: #333333;
                border: 1px solid #cccccc;
                padding: 5px;
                border-radius: 3px;
            }
            QTabBar::tab:selected {
                background: #e0e0e0;
            }
            QTableWidget {
                background-color: #ffffff;
                color: #333333;
                alternate-background-color: #f5f5f5;
                border: 1px solid #cccccc;
            }
            QGroupBox {
                border: 1px solid #cccccc;
                color: #333333;
                margin-top: 10px;
                border-radius: 5px;
            }
        """

      
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        
        self.create_header_section(main_layout)
        self.create_content_section(main_layout)
        
       
        self.load_initial_data()
        
        
        self.setup_refresh_timer()

    def create_header_section(self, parent_layout):
        
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins for a compact layout

        
        select_label = QLabel("Select Patient:")
        select_label.setFont(QFont("Arial", 10, QFont.Bold))  # Smaller font for compactness
        self.patient_combo = QComboBox()
        self.patient_combo.setMinimumWidth(250)  # Reduce width slightly
        self.patient_combo.currentIndexChanged.connect(self.on_patient_select)

      
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        search_label.setFont(QFont("Arial", 10))  # Match font size with other widgets
        self.search_input = QLineEdit()
        self.search_input.textChanged.connect(self.filter_patients)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_input)

        self.patient_info = QLabel()
        self.patient_info.setFont(QFont("Arial", 10))

       
        self.dark_mode_button = QPushButton("Enable Dark Mode")
        self.dark_mode_button.setFont(QFont("Arial", 10))  # Smaller font to match header
        self.dark_mode_button.clicked.connect(self.toggle_dark_mode)

        
        icon_container = QWidget()
        icon_layout = QVBoxLayout(icon_container)
        icon_layout.setAlignment(Qt.AlignCenter)
        icon_layout.setContentsMargins(0, 0, 0, 0)  # Tighten the layout around the icon

        
        icon_label = QLabel()
        icon_pixmap = QPixmap(os.path.join(os.path.dirname(__file__), "metis_icon.png"))
        if not icon_pixmap.isNull():
            icon_label.setPixmap(
                icon_pixmap.scaled(45, 45, Qt.KeepAspectRatio, Qt.SmoothTransformation)  # 3/4 of original size
            )
        icon_layout.addWidget(icon_label)

        
        greek_font_path = os.path.join(os.path.dirname(__file__), "..", "..", "fonts", "CinzelDecorative-Regular.ttf")
        greek_font_id = QFontDatabase.addApplicationFont(greek_font_path)
        greek_font_family = QFontDatabase.applicationFontFamilies(greek_font_id)[0] if greek_font_id != -1 else "Arial"
        greek_font = QFont(greek_font_family, 10, QFont.Bold)  

        metis_label = QLabel("METIS")  
        metis_label.setFont(greek_font)
        metis_label.setAlignment(Qt.AlignCenter)
        icon_layout.addWidget(metis_label)

       
        header_layout.addWidget(select_label)
        header_layout.addWidget(self.patient_combo)
        header_layout.addLayout(search_layout)
        header_layout.addWidget(self.patient_info)
        header_layout.addWidget(self.dark_mode_button)

        
        spacer = QWidget()
        spacer.setFixedWidth(10) 
        header_layout.addWidget(spacer)

       
        header_layout.addWidget(icon_container)

        parent_layout.addWidget(header)




    def toggle_dark_mode(self):
        
        self.is_dark_mode = not self.is_dark_mode
        if self.is_dark_mode:
            self.setStyleSheet(self.dark_mode_stylesheet)
            self.dark_mode_button.setText("Enable Light Mode")
        else:
            self.setStyleSheet(self.light_mode_stylesheet)
            self.dark_mode_button.setText("Enable Dark Mode")

    def create_content_section(self, parent_layout):
        
        self.tab_widget = QTabWidget()
        
        # Create main tabs
        self.create_overview_tab()
        self.create_medications_tab()
        self.create_analytics_tab()
        self.create_ml_risk_tab()
        
        parent_layout.addWidget(self.tab_widget)

    def create_overview_tab(self):
        
        self.overview_tab = PatientOverviewTab(self)
        self.tab_widget.addTab(self.overview_tab, "Patient Overview")

    def create_medications_tab(self):
        
        self.medications_tab = MedicationsTab(self)
        self.medications_tab.order_button.clicked.connect(self.place_order)
        self.tab_widget.addTab(self.medications_tab, "Medications")

    def create_analytics_tab(self):
        
        self.analytics_tab = PatientAnalyticsTab(self)
        self.tab_widget.addTab(self.analytics_tab, "Trends")

    def create_ml_risk_tab(self):
        
        self.ml_risk_tab = MLRiskTab(self)
        self.tab_widget.addTab(self.ml_risk_tab, "ML Risk Analysis")

    def load_initial_data(self):
        
        self.load_patients()
        self.load_medications()
        self.load_ranges()

    def setup_refresh_timer(self):
        
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_vitals)
        self.refresh_timer.start(60000)  # Refresh every minute

    def closeEvent(self, event):
        
        self.refresh_timer.stop()
        self.cdss.close()
        super().closeEvent(event)
