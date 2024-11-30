import sys
import os

# Add the current directory to Python path
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

# Import all necessary components
from UI.main_window.base import CDSSUI
from PySide6.QtWidgets import QApplication
from cdss_functions import CDSSSystem
from ml.ml_risk_predictor import EnhancedRiskPredictor

def main():
    try:
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        window = CDSSUI()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()