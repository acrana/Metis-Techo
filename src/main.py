# src/main.py

from risk_assessment import get_patient_risk_assessment
from data_access import get_patient_demographics, get_patient_surveys, get_patient_ade_records

def main():
    while True:
        print("\nClinical Decision Support System")
        print("1. View Patient Demographics")
        print("2. View Patient Surveys")
        print("3. View Patient ADE Records")
        print("4. Assess Patient Risk")
        print("5. Exit")
        choice = input("Enter choice: ")

        if choice == '1':
            patient_id = input("Enter Patient ID: ")
            demo = get_patient_demographics(int(patient_id))
            print("\nPatient Demographics:")
            print(demo)
        elif choice == '2':
            patient_id = input("Enter Patient ID: ")
            surveys = get_patient_surveys(int(patient_id))
            print("\nPatient Surveys:")
            for survey in surveys:
                print(survey)
        elif choice == '3':
            patient_id = input("Enter Patient ID: ")
            ade_records = get_patient_ade_records(int(patient_id))
            print("\nPatient ADE Records:")
            for record in ade_records:
                print(record)
        elif choice == '4':
            patient_id = input("Enter Patient ID: ")
            get_patient_risk_assessment(int(patient_id))
        elif choice == '5':
            print("Exiting the application.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    main()
