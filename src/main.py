from data_access import get_patient_demographics, get_patient_surveys, get_patient_ade_records
from risk_assessment import get_patient_risk_assessment

def main():
    while True:
        print("""
Clinical Decision Support System
1. View Patient Demographics
2. View Patient Surveys
3. View Patient ADE Records
4. Assess Patient Risk
5. Exit
Enter choice:""")
        choice = input().strip()

        if choice == '1':
            patient_id = input("Enter Patient ID: ").strip()
            demographics = get_patient_demographics(patient_id)
            if demographics.empty:
                print(f"No demographics found for Patient ID {patient_id}.")
            else:
                print(demographics)
        elif choice == '2':
            patient_id = input("Enter Patient ID: ").strip()
            surveys = get_patient_surveys(patient_id)
            if surveys.empty:
                print(f"No surveys found for Patient ID {patient_id}.")
            else:
                print(surveys)
        elif choice == '3':
            patient_id = input("Enter Patient ID: ").strip()
            ade_records = get_patient_ade_records(patient_id)
            if ade_records.empty:
                print(f"No ADE records found for Patient ID {patient_id}.")
            else:
                print(ade_records)
        elif choice == '4':
            patient_id = input("Enter Patient ID: ").strip()
            assessment = get_patient_risk_assessment(patient_id)
            if assessment is None:
                print("Risk assessment could not be performed.")
            else:
                print(f"""
Patient ID: {assessment['PatientID']}
Predicted ADE Risk: {assessment['RiskLevel']}
Risk Probability: {assessment['RiskProbability']:.2f}
""")
        elif choice == '5':
            print("Exiting the application.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    main()
