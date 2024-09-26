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
}

# Define lower normal limits for lab tests where necessary
lower_limits_normal = {
    'TSH': 0.4,                # Lower limit for TSH
    'Platelets': 150,          # Lower limit for Platelets
    'Sodium': 135,             # Lower limit for Sodium
    'Blood Pressure': 90,      # Lower limit for Systolic Blood Pressure
    'Blood Glucose': 70,       # Lower limit for fasting Blood Glucose
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
        'Interactions': ['Spironolactone', 'NSAIDs', 'Aliskiren']  # Hyperkalemia risk with Spironolactone; NSAIDs reduce efficacy
    },
    'Metformin': {
        'SideEffects': ['Nausea', 'Diarrhea', 'Lactic Acidosis'],
        'Contraindications': {
            'Creatinine': lambda x: x > 1.5,
            'Allergies': ['Metformin'],
        },
        'MonitoringParameters': ['Blood Glucose', 'Creatinine'],
        'Interactions': ['Cimetidine', 'Digoxin']  # Cimetidine can increase Metformin levels; Digoxin interaction may increase risk of lactic acidosis
    },
    'Amiodarone': {
        'SideEffects': ['Thyroid Dysfunction', 'Pulmonary Toxicity', 'Liver Toxicity'],
        'Contraindications': {
            'QTc': lambda x: x > upper_limits_normal['QTc'],
            'Allergies': ['Amiodarone'],
        },
        'MonitoringParameters': ['QTc', 'Liver Enzymes', 'Thyroid Function'],
        'Interactions': ['Digoxin', 'Warfarin', 'Simvastatin', 'Beta-Blockers']  # Amiodarone increases Digoxin and Warfarin levels; QT prolongation with Beta-blockers
    },
    'Ibuprofen': {
        'SideEffects': ['Gastrointestinal Bleeding', 'Kidney Dysfunction'],
        'Contraindications': {
            'Creatinine': lambda x: x > 1.8,
            'Allergies': ['Ibuprofen', 'NSAIDs'],
            'Gastrointestinal Issues': True,  # If patient has GI issues
        },
        'MonitoringParameters': ['Creatinine'],
        'Interactions': ['ACE Inhibitors', 'Aspirin', 'Warfarin']  # Increased risk of kidney damage with ACE inhibitors; bleeding with Warfarin or Aspirin
    },
    'Warfarin': {
        'SideEffects': ['Bleeding', 'Bruising'],
        'Contraindications': {
            'INR': lambda x: x > 3.0,
            'Platelets': lambda x: x < lower_limits_normal['Platelets'],
            'Allergies': ['Warfarin'],
        },
        'MonitoringParameters': ['INR', 'Platelets'],
        'Interactions': ['Amiodarone', 'Aspirin', 'NSAIDs', 'Antibiotics']  # Amiodarone increases Warfarin levels; Aspirin increases bleeding risk
    },
    'Insulin': {
        'SideEffects': ['Hypoglycemia', 'Weight Gain'],
        'Contraindications': {
            'Blood Glucose': lambda x: x < lower_limits_normal['Blood Glucose'],
            'Allergies': ['Insulin'],
        },
        'MonitoringParameters': ['Blood Glucose'],
        'Interactions': ['Beta-Blockers', 'Thiazides', 'Corticosteroids']  # Beta-blockers mask hypoglycemia; Thiazides and Corticosteroids increase blood sugar
    },
    'Atorvastatin': {
        'SideEffects': ['Muscle Pain', 'Liver Enzyme Elevation'],
        'Contraindications': {
            'Liver Enzymes': lambda x: x > 3 * upper_limits_normal['Liver Enzymes'],
            'Allergies': ['Atorvastatin'],
        },
        'MonitoringParameters': ['Liver Enzymes'],
        'Interactions': ['Amiodarone', 'Grapefruit Juice', 'Warfarin']  # Increased muscle toxicity with Amiodarone; grapefruit juice increases levels
    },
    'Aspirin': {
        'SideEffects': ['Bleeding', 'Gastric Ulcer'],
        'Contraindications': {
            'Platelets': lambda x: x < lower_limits_normal['Platelets'],
            'Allergies': ['Aspirin', 'NSAIDs'],
            'Gastrointestinal Issues': True,
        },
        'MonitoringParameters': ['Platelets'],
        'Interactions': ['Warfarin', 'Ibuprofen', 'Corticosteroids']  # Increased bleeding risk with Warfarin; Ibuprofen reduces aspirin's antiplatelet effect
    },
    'Levothyroxine': {
        'SideEffects': ['Palpitations', 'Insomnia', 'Heat Intolerance'],
        'Contraindications': {
            'TSH': lambda x: x < lower_limits_normal['TSH'],
            'Allergies': ['Levothyroxine'],
        },
        'MonitoringParameters': ['TSH', 'T4'],
        'Interactions': ['Calcium Supplements', 'Iron Supplements']  # Reduced absorption with calcium and iron supplements
    },
    'Hydrochlorothiazide': {
        'SideEffects': ['Electrolyte Imbalance', 'Dehydration'],
        'Contraindications': {
            'Sodium': lambda x: x < lower_limits_normal['Sodium'],
            'Allergies': ['Sulfonamides'],
        },
        'MonitoringParameters': ['Electrolytes', 'Blood Pressure'],
        'Interactions': ['ACE Inhibitors', 'NSAIDs', 'Lithium']  # ACE inhibitors and thiazides can cause low sodium; Lithium toxicity risk
    },
    'Clopidogrel': {
        'SideEffects': ['Bleeding', 'Bruising'],
        'Contraindications': {
            'Platelets': lambda x: x < lower_limits_normal['Platelets'],
            'Allergies': ['Clopidogrel'],
        },
        'MonitoringParameters': ['Platelets'],
        'Interactions': ['Proton Pump Inhibitors', 'Aspirin', 'NSAIDs']  # PPIs can reduce efficacy; NSAIDs and Aspirin increase bleeding risk
    },
    'Digoxin': {
        'SideEffects': ['Nausea', 'Arrhythmias', 'Visual Disturbances'],
        'Contraindications': {
            'Digoxin Level': lambda x: x > upper_limits_normal['Digoxin Level'],
            'Allergies': ['Digoxin'],
        },
        'MonitoringParameters': ['Digoxin Level', 'Electrolytes'],
        'Interactions': ['Amiodarone', 'Verapamil', 'Diuretics']  # Amiodarone and Verapamil increase Digoxin levels; Diuretics may lead to electrolyte imbalances
    },
    'Losartan': {
        'SideEffects': ['Dizziness', 'Hyperkalemia'],
        'Contraindications': {
            'Potassium': lambda x: x > upper_limits_normal['Potassium'],
            'Allergies': ['Losartan', 'ARBs'],
        },
        'MonitoringParameters': ['Blood Pressure', 'Potassium'],
        'Interactions': ['Spironolactone', 'NSAIDs', 'Lithium']  # Hyperkalemia risk with Spironolactone; reduced effect with NSAIDs; Lithium toxicity
    },
    'Gabapentin': {
        'SideEffects': ['Drowsiness', 'Dizziness'],
        'Contraindications': {
            'Allergies': ['Gabapentin'],
        },
        'MonitoringParameters': ['Renal Function'],
        'Interactions': ['Antacids', 'Opioids']  # Antacids reduce absorption; opioids increase sedation
    },
    'Prednisone': {
        'SideEffects': ['Immunosuppression', 'Hyperglycemia'],
        'Contraindications': {
            'Blood Glucose': lambda x: x > 250,
            'Allergies': ['Prednisone'],
        },
        'MonitoringParameters': ['Blood Glucose'],
        'Interactions': ['NSAIDs', 'Warfarin', 'Insulin']  # Increased ulcer risk with NSAIDs; altered Warfarin levels; raises blood sugar, affecting Insulin
    },
    'Omeprazole': {
        'SideEffects': ['Headache', 'Diarrhea'],
        'Contraindications': {
            'Allergies': ['Omeprazole'],
        },
        'MonitoringParameters': [],
        'Interactions': ['Clopidogrel', 'Warfarin']  # Reduces efficacy of Clopidogrel; affects Warfarin metabolism
    },
    'Amlodipine': {
        'SideEffects': ['Edema', 'Dizziness'],
        'Contraindications': {
            'Blood Pressure': lambda x: x < lower_limits_normal['Blood Pressure'],
            'Allergies': ['Amlodipine'],
        },
        'MonitoringParameters': ['Blood Pressure'],
        'Interactions': ['Simvastatin', 'Beta-Blockers']  # Simvastatin levels are increased; hypotension with Beta-Blockers
    },
}

