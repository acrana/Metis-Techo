from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QComboBox, QTableWidget, QTableWidgetItem
)
import pyqtgraph as pg
from datetime import datetime


class PatientAnalyticsTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

       
        self.parameter_selector = QComboBox()
        self.parameter_selector.addItems([
            "Select Parameter", "Heart Rate", "Blood Pressure", "Potassium", "QTc",
            "CBC Platelets", "Hemoglobin", "Blood Glucose"
        ])
        self.parameter_selector.currentTextChanged.connect(self.on_parameter_change)
        layout.addWidget(self.parameter_selector)

        # Group for Analytics
        analytics_group = QGroupBox("Analytics")
        analytics_layout = QVBoxLayout(analytics_group)

        # Plot Widget
        self.analytics_plot = pg.PlotWidget()
        self.analytics_plot.setFixedHeight(200)
        self.analytics_plot.getPlotItem().getAxis("bottom").setTickSpacing(1, 1)
        self.analytics_plot.getPlotItem().getAxis("bottom").setStyle(showValues=True)
        self.analytics_plot.getPlotItem().getAxis("bottom").setLabel("Time")
        analytics_layout.addWidget(self.analytics_plot)

        # Table Widget
        self.analytics_table = QTableWidget(0, 2)
        self.analytics_table.setHorizontalHeaderLabels(["Time", "Value"])
        analytics_layout.addWidget(self.analytics_table)

        layout.addWidget(analytics_group)
        self.setLayout(layout)

    def on_parameter_change(self, parameter):
        
        if parameter == "Select Parameter" or not self.main_window.current_patient_id:
            self.clear_display()
            return

        if parameter in ["Heart Rate", "Blood Pressure"]:
            self.update_vital_trend(parameter)
        elif parameter in ["Potassium", "QTc", "CBC Platelets", "Hemoglobin", "Blood Glucose"]:
            self.update_lab_trend(parameter)

    def update_analytics(self, patient_id):
        
        self.on_parameter_change(self.parameter_selector.currentText())

    def update_vital_trend(self, parameter):
        
        cursor = self.main_window.cdss.conn.cursor()

        if parameter == "Heart Rate":
            query = """
                SELECT timestamp, heart_rate
                FROM Vitals
                WHERE patient_id = ?
                ORDER BY timestamp
            """
        elif parameter == "Blood Pressure":
            query = """
                SELECT timestamp, blood_pressure_systolic, blood_pressure_diastolic
                FROM Vitals
                WHERE patient_id = ?
                ORDER BY timestamp
            """

        cursor.execute(query, (self.main_window.current_patient_id,))
        vitals = cursor.fetchall()

        if not vitals:
            self.clear_display()
            return

        self.analytics_plot.clear()
        self.analytics_table.setRowCount(0)

        times = []
        values = []

        for i, vital in enumerate(vitals):
            timestamp = datetime.fromisoformat(vital["timestamp"]).isoformat()
            times.append(datetime.fromisoformat(vital["timestamp"]).timestamp())

            if parameter == "Heart Rate":
                value = float(vital["heart_rate"])
                values.append(value)
            elif parameter == "Blood Pressure":
                value = f"{vital['blood_pressure_systolic']}/{vital['blood_pressure_diastolic']}"
                values.append(float(vital["blood_pressure_systolic"]))

            self.analytics_table.insertRow(i)
            self.analytics_table.setItem(i, 0, QTableWidgetItem(timestamp))
            self.analytics_table.setItem(i, 1, QTableWidgetItem(str(value)))

        self.analytics_plot.plot(times, values, pen="r", name=parameter)
        self.analytics_plot.getPlotItem().getAxis("bottom").setTicks([list(zip(times, [datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S') for t in times]))])

    def update_lab_trend(self, parameter):
        
        cursor = self.main_window.cdss.conn.cursor()

        lab_name_mapping = {
            "Potassium": "Potassium",
            "QTc": "QTc",
            "CBC Platelets": "CBC Platelets",
            "Hemoglobin": "Hemoglobin",
            "Blood Glucose": "Blood Glucose"
        }

        query = """
            SELECT lr.timestamp, lr.result
            FROM Lab_Results lr
            JOIN Labs l ON lr.lab_id = l.lab_id
            WHERE lr.patient_id = ? AND l.lab_name = ?
            ORDER BY lr.timestamp
        """

        cursor.execute(query, (self.main_window.current_patient_id, lab_name_mapping[parameter]))
        labs = cursor.fetchall()

        if not labs:
            self.clear_display()
            return

        self.analytics_plot.clear()
        self.analytics_table.setRowCount(0)

        times = []
        values = []

        for i, lab in enumerate(labs):
            timestamp = datetime.fromisoformat(lab["timestamp"]).isoformat()
            value = float(lab["result"])
            times.append(datetime.fromisoformat(lab["timestamp"]).timestamp())
            values.append(value)

            self.analytics_table.insertRow(i)
            self.analytics_table.setItem(i, 0, QTableWidgetItem(timestamp))
            self.analytics_table.setItem(i, 1, QTableWidgetItem(str(value)))

        self.analytics_plot.plot(times, values, pen="g", name=parameter)
        self.analytics_plot.getPlotItem().getAxis("bottom").setTicks([list(zip(times, [datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S') for t in times]))])

    def clear_display(self):
        
        self.analytics_plot.clear()
        self.analytics_table.setRowCount(0)