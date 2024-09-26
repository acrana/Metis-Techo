# medications.py

medications = {
    # Cardiovascular Medications
    'Lisinopril': {
        'SideEffects': ['Cough', 'Dizziness', 'Hyperkalemia'],
        'Contraindications': {
            'Creatinine': lambda x: x > 2.0,  # Risk of renal impairment
            'Potassium': lambda x: x > 5.0,   # Risk of hyperkalemia
            'Allergies': ['ACE Inhibitors'],
        },
        'MonitoringParameters': ['Blood Pressure', 'Creatinine', 'Potassium'],
    },
    'Amlodipine': {
        'SideEffects': ['Peripheral Edema', 'Dizziness', 'Fatigue'],
        'Contraindications': {
            'Blood Pressure': lambda x: x < 90,  # Hypotension risk
        },
        'MonitoringParameters': ['Blood Pressure', 'Heart Rate'],
    },
    'Clonidine': {
        'SideEffects': ['Dry Mouth', 'Drowsiness', 'Bradycardia'],
        'Contraindications': {
            'HeartRate': lambda x: x < 60,  # Risk of bradycardia
        },
        'MonitoringParameters': ['Blood Pressure', 'Heart Rate'],
    },
    
    # Antipsychotics
    'Quetiapine (Seroquel)': {
        'SideEffects': ['Sedation', 'Weight Gain', 'QT Prolongation'],
        'Contraindications': {
            'QTc': lambda x: x > 450,  # Risk of QT prolongation and arrhythmias
            'Diabetes': lambda x: x == True,  # Risk of worsening glucose control
            'Allergies': ['Quetiapine'],
        },
        'MonitoringParameters': ['Blood Glucose', 'QTc', 'Lipid Profile', 'Weight'],
    },
    'Olanzapine (Zyprexa)': {
        'SideEffects': ['Weight Gain', 'Hyperglycemia', 'Orthostatic Hypotension'],
        'Contraindications': {
            'Diabetes': lambda x: x == True,  # Worsens glucose control
            'Allergies': ['Olanzapine'],
        },
        'MonitoringParameters': ['Blood Glucose', 'Lipid Profile', 'Weight'],
    },
    'Haloperidol (Haldol)': {
        'SideEffects': ['Extrapyramidal Symptoms', 'QT Prolongation', 'Sedation'],
        'Contraindications': {
            'QTc': lambda x: x > 470,  # QT prolongation risk
            'Allergies': ['Haloperidol'],
        },
        'MonitoringParameters': ['QTc', 'Extrapyramidal Symptoms (EPS)'],
    },

    # Mood Stabilizers and Antiepileptics
    'Lithium': {
        'SideEffects': ['Tremor', 'Hypothyroidism', 'Nephrogenic Diabetes Insipidus'],
        'Contraindications': {
            'Creatinine': lambda x: x > 1.5,  # Renal impairment increases toxicity
            'Sodium': lambda x: x < 135,  # Low sodium increases lithium toxicity risk
            'Allergies': ['Lithium'],
        },
        'MonitoringParameters': ['Lithium Levels', 'Creatinine', 'TSH', 'Sodium'],
    },
    'Valproic Acid (Depakote)': {
        'SideEffects': ['Hepatotoxicity', 'Thrombocytopenia', 'Weight Gain'],
        'Contraindications': {
            'Liver Enzymes': lambda x: x > 2 * 40,  # Hepatotoxicity risk
            'Platelets': lambda x: x < 100,  # Thrombocytopenia risk
            'Allergies': ['Valproic Acid'],
        },
        'MonitoringParameters': ['Liver Enzymes', 'Platelet Count', 'Valproic Acid Levels'],
    },
    'Lamotrigine (Lamictal)': {
        'SideEffects': ['Rash', 'Stevens-Johnson Syndrome', 'Dizziness'],
        'Contraindications': {
            'Allergies': ['Lamotrigine'],
        },
        'MonitoringParameters': ['Skin Rash (SJS)', 'CNS Symptoms'],
    },
    
    # Diabetic Medications
    'Metformin': {
        'SideEffects': ['Nausea', 'Lactic Acidosis', 'Diarrhea'],
        'Contraindications': {
            'Creatinine': lambda x: x > 1.5,  # Avoid in patients with renal impairment
            'Liver Enzymes': lambda x: x > 3 * 40,  # Hepatic dysfunction risk
        },
        'MonitoringParameters': ['Blood Glucose', 'Creatinine'],
    },
    'Insulin (Subcutaneous)': {
        'SideEffects': ['Hypoglycemia', 'Weight Gain'],
        'Contraindications': {
            'Blood Glucose': lambda x: x < 70,  # Risk of hypoglycemia
        },
        'MonitoringParameters': ['Blood Glucose'],
    },
    
    # Anticoagulants
    'Apixaban (Eliquis)': {
        'SideEffects': ['Bleeding', 'Anemia'],
        'Contraindications': {
            'Platelets': lambda x: x < 100,  # Risk of bleeding
            'INR': lambda x: x > 3.0,  # High bleeding risk
            'Allergies': ['Apixaban'],
        },
        'MonitoringParameters': ['INR', 'Platelet Count'],
    },
    'Rivaroxaban (Xarelto)': {
        'SideEffects': ['Bleeding', 'Hematoma'],
        'Contraindications': {
            'Platelets': lambda x: x < 100,  # Risk of bleeding
            'INR': lambda x: x > 3.0,  # Risk of excessive anticoagulation
        },
        'MonitoringParameters': ['INR', 'Platelet Count'],
    },
    
    # Antibiotics
    'Piperacillin/Tazobactam (Zosyn)': {
        'SideEffects': ['Rash', 'Thrombocytopenia', 'Diarrhea'],
        'Contraindications': {
            'Allergies': ['Penicillin'],
            'Creatinine': lambda x: x > 2.0,  # Dose adjustment needed for renal impairment
        },
        'MonitoringParameters': ['Renal Function', 'Platelets'],
    },
    'Ceftriaxone (Rocephin)': {
        'SideEffects': ['Diarrhea', 'Cholestasis', 'Rash'],
        'Contraindications': {
            'Allergies': ['Cephalosporins'],
            'Liver Enzymes': lambda x: x > 3 * 40,  # Risk of cholestasis
        },
        'MonitoringParameters': ['Liver Enzymes', 'Renal Function'],
    },
    
    # Immunosuppressants
    'Tacrolimus': {
        'SideEffects': ['Nephrotoxicity', 'Hyperglycemia', 'Hypertension'],
        'Contraindications': {
            'Creatinine': lambda x: x > 1.5,  # Risk of nephrotoxicity
            'Blood Pressure': lambda x: x > 160,  # Hypertension risk
        },
        'MonitoringParameters': ['Creatinine', 'Blood Pressure', 'Blood Glucose'],
    },
    'Mycophenolate Mofetil (Cellcept)': {
        'SideEffects': ['Diarrhea', 'Leukopenia', 'Infections'],
        'Contraindications': {
            'Platelets': lambda x: x < 50,  # Risk of leukopenia and bleeding
        },
        'MonitoringParameters': ['CBC', 'Platelets', 'Liver Enzymes'],
    },

    # Pain Management
    'Gabapentin (Neurontin)': {
        'SideEffects': ['Drowsiness', 'Dizziness', 'Peripheral Edema'],
        'Contraindications': {
            'Creatinine': lambda x: x > 2.0,  # Dose adjustment required for renal impairment
        },
        'MonitoringParameters': ['Renal Function'],
    },
    'Acetaminophen (Tylenol)': {
        'SideEffects': ['Hepatotoxicity', 'Nausea', 'Rash'],
        'Contraindications': {
            'Liver Enzymes': lambda x: x > 3 * 40,  # Avoid in hepatic dysfunction
        },
        'MonitoringParameters': ['Liver Enzymes'],
    },
    
    # Thyroid Medications
    'Levothyroxine (Synthroid)': {
        'SideEffects': ['Palpitations', 'Insomnia', 'Weight Loss'],
        'Contraindications': {
            'TSH': lambda x: x < 0.3,  # Risk of over-suppression
            'Allergies': ['Levothyroxine'],
        },
        'MonitoringParameters': ['TSH', 'T4'],
    }
}

