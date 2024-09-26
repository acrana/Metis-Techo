# medications.py

# Define upper normal limits for various lab tests
upper_limits_normal = {
    'Liver Enzymes': 40,       # Upper limit for ALT/AST
    'TSH': 4.0,                # Upper limit for Thyroid-Stimulating Hormone
    'Potassium': 5.0,          # Upper limit for Potassium
    'INR': 1.1,                # Upper limit for International Normalized Ratio
    'Creatinine': 1.2,         # Upper limit for Creatinine
    'Blood Glucose': 99,       # Upper limit for fasting Blood Glucose
    'QTc': 450,                # Upper limit for QTc interval in milliseconds
    'Platelets': 450,          # Upper limit for Platelet count (x10^9/L)
    'Sodium': 145,             # Upper limit for Sodium
    'Digoxin Level': 2.0,      # Therapeutic upper limit for Digoxin
    'Blood Pressure': 120,     # Upper limit for Systolic Blood Pressure
    # Add other lab upper limits as needed
}

# Define lower normal limits for lab tests where necessary
lower_limits_normal = {
    'TSH': 0.4,                # Lower limit for TSH
    'Platelets': 150,          # Lower limit for Platelets
    'Sodium': 135,             # Lower limit for Sodium
    'Blood Pressure': 90,      # Lower limit for Systolic Blood Pressure
    'Blood Glucose': 70,       # Lower limit for fasting Blood Glucose
    # Add other lab lower limits as needed
}

medications = {
    'Lisinopril': {
        'SideEffects': ['Cough', 'Dizziness', 'Hyperkalemia'],
        'Contraindications': {
            'Creatinine': lambda x: x > 2.0,
            'Potassium': lambda x: x > upper_limits_normal['Potassium'],
            'Allergies': ['Lisinopril', 'ACE Inhibitors'],
        },
        'MonitoringParameters': ['Blood Pressure', 'Creatinine', 'Potassium'],
    },
    'Metformin': {
        'SideEffects': ['Nausea', 'Diarrhea', 'Lactic Acidosis'],
        'Contraindications': {
            'Creatinine': lambda x: x > 1.5,
            'Allergies': ['Metformin'],
        },
        'MonitoringParameters': ['Blood Glucose', 'Creatinine'],
    },
    'Amiodarone': {
        'SideEffects': ['Thyroid Dysfunction', 'Pulmonary Toxicity', 'Liver Toxicity'],
        'Contraindications': {
            'QTc': lambda x: x > upper_limits_normal['QTc'],
            'Allergies': ['Amiodarone'],
        },
        'MonitoringParameters': ['QTc', 'Liver Enzymes', 'Thyroid Function'],
    },
    'Ibuprofen': {
        'SideEffects': ['Gastrointestinal Bleeding', 'Kidney Dysfunction'],
        'Contraindications': {
            'Creatinine': lambda x: x > 1.8,
            'Allergies': ['Ibuprofen', 'NSAIDs'],
            'Gastrointestinal Issues': True,  # If patient has GI issues
        },
        'MonitoringParameters': ['Creatinine'],
    },
    'Warfarin': {
        'SideEffects': ['Bleeding', 'Bruising'],
        'Contraindications': {
            'INR': lambda x: x > 3.0,
            'Platelets': lambda x: x < lower_limits_normal['Platelets'],
            'Allergies': ['Warfarin'],
        },
        'MonitoringParameters': ['INR', 'Platelets'],
    },
    'Insulin': {
        'SideEffects': ['Hypoglycemia', 'Weight Gain'],
        'Contraindications': {
            'Blood Glucose': lambda x: x < lower_limits_normal['Blood Glucose'],
            'Allergies': ['Insulin'],
        },
        'MonitoringParameters': ['Blood Glucose'],
    },
    'Atorvastatin': {
        'SideEffects': ['Muscle Pain', 'Liver Enzyme Elevation'],
        'Contraindications': {
            'Liver Enzymes': lambda x: x > 3 * upper_limits_normal['Liver Enzymes'],
            'Allergies': ['Atorvastatin'],
        },
        'MonitoringParameters': ['Liver Enzymes'],
    },
    'Aspirin': {
        'SideEffects': ['Bleeding', 'Gastric Ulcer'],
        'Contraindications': {
            'Platelets': lambda x: x < lower_limits_normal['Platelets'],
            'Allergies': ['Aspirin', 'NSAIDs'],
            'Gastrointestinal Issues': True,
        },
        'MonitoringParameters': ['Platelets'],
    },
    'Levothyroxine': {
        'SideEffects': ['Palpitations', 'Insomnia', 'Heat Intolerance'],
        'Contraindications': {
            'TSH': lambda x: x < lower_limits_normal['TSH'],
            'Allergies': ['Levothyroxine'],
        },
        'MonitoringParameters': ['TSH', 'T4'],
    },
    'Hydrochlorothiazide': {
        'SideEffects': ['Electrolyte Imbalance', 'Dehydration'],
        'Contraindications': {
            'Sodium': lambda x: x < lower_limits_normal['Sodium'],
            'Allergies': ['Sulfonamides'],
        },
        'MonitoringParameters': ['Electrolytes', 'Blood Pressure'],
    },
    'Clopidogrel': {
        'SideEffects': ['Bleeding', 'Bruising'],
        'Contraindications': {
            'Platelets': lambda x: x < lower_limits_normal['Platelets'],
            'Allergies': ['Clopidogrel'],
        },
        'MonitoringParameters': ['Platelets'],
    },
    'Digoxin': {
        'SideEffects': ['Nausea', 'Arrhythmias', 'Visual Disturbances'],
        'Contraindications': {
            'Digoxin Level': lambda x: x > upper_limits_normal['Digoxin Level'],
            'Allergies': ['Digoxin'],
        },
        'MonitoringParameters': ['Digoxin Level', 'Electrolytes'],
    },
    'Losartan': {
        'SideEffects': ['Dizziness', 'Hyperkalemia'],
        'Contraindications': {
            'Potassium': lambda x: x > upper_limits_normal['Potassium'],
            'Allergies': ['Losartan', 'ARBs'],
        },
        'MonitoringParameters': ['Blood Pressure', 'Potassium'],
    },
    'Gabapentin': {
        'SideEffects': ['Drowsiness', 'Dizziness'],
        'Contraindications': {
            'Allergies': ['Gabapentin'],
        },
        'MonitoringParameters': ['Renal Function'],
    },
    'Prednisone': {
        'SideEffects': ['Immunosuppression', 'Hyperglycemia'],
        'Contraindications': {
            'Blood Glucose': lambda x: x > 250,
            'Allergies': ['Prednisone'],
        },
        'MonitoringParameters': ['Blood Glucose'],
    },
    'Omeprazole': {
        'SideEffects': ['Headache', 'Diarrhea'],
        'Contraindications': {
            'Allergies': ['Omeprazole'],
        },
        'MonitoringParameters': [],
    },
    'Amlodipine': {
        'SideEffects': ['Edema', 'Dizziness'],
        'Contraindications': {
            'Blood Pressure': lambda x: x < lower_limits_normal['Blood Pressure'],
            'Allergies': ['Amlodipine'],
        },
        'MonitoringParameters': ['Blood Pressure'],
    },
  
}
