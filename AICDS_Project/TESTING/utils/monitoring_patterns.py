MEDICATION_MONITORING = {
    "Metoprolol": {
        "class": "Beta Blocker",
        "monitoring_parameters": {
            "heart_rate": {
                "baseline_range": (80, 100),
                "therapeutic_range": (60, 80),
                "adverse_threshold_low": 50,
                "adverse_threshold_high": 120,
                "monitoring_frequency": "daily",
                "check_duration": 14,  # days to monitor after starting
                "expected_effect": "decrease",
                "critical_values": {
                    "low": 45,
                    "high": 130
                }
            },
            "blood_pressure_systolic": {
                "baseline_range": (140, 160),
                "therapeutic_range": (120, 140),
                "adverse_threshold_low": 90,
                "adverse_threshold_high": 180,
                "monitoring_frequency": "daily",
                "check_duration": 14,
                "expected_effect": "decrease",
                "critical_values": {
                    "low": 80,
                    "high": 200
                }
            },
            "blood_pressure_diastolic": {
                "baseline_range": (90, 100),
                "therapeutic_range": (70, 90),
                "adverse_threshold_low": 60,
                "adverse_threshold_high": 110,
                "monitoring_frequency": "daily",
                "check_duration": 14,
                "expected_effect": "decrease",
                "critical_values": {
                    "low": 50,
                    "high": 120
                }
            }
        },
        "success_pattern": {
            "timeline": {
                "days_0_3": "gradual_decrease",
                "days_4_7": "stabilize_therapeutic",
                "days_8_plus": "maintain_therapeutic"
            }
        },
        "adverse_patterns": [
            {
                "name": "excessive_lowering",
                "probability": 0.15,
                "timeline": {
                    "days_0_2": "rapid_decrease",
                    "days_3_plus": "below_threshold"
                }
            },
            {
                "name": "ineffective",
                "probability": 0.10,
                "timeline": {
                    "days_0_7": "minimal_change",
                    "days_8_plus": "above_therapeutic"
                }
            }
        ]
    },

    "Lisinopril": {
        "class": "ACE Inhibitor",
        "monitoring_parameters": {
            "blood_pressure_systolic": {
                "baseline_range": (150, 170),
                "therapeutic_range": (120, 140),
                "adverse_threshold_low": 90,
                "adverse_threshold_high": 180,
                "monitoring_frequency": "daily",
                "check_duration": 14,
                "expected_effect": "decrease",
                "critical_values": {
                    "low": 80,
                    "high": 200
                }
            },
            "creatinine": {
                "baseline_range": (0.7, 1.2),
                "therapeutic_range": (0.7, 1.4),  # Allow 30% increase
                "adverse_threshold_high": 1.6,
                "monitoring_frequency": "weekly",
                "check_duration": 28,
                "expected_effect": "stable",
                "critical_values": {
                    "high": 2.0
                }
            },
            "potassium": {
                "baseline_range": (3.5, 5.0),
                "therapeutic_range": (3.5, 5.0),
                "adverse_threshold_low": 3.0,
                "adverse_threshold_high": 5.5,
                "monitoring_frequency": "weekly",
                "check_duration": 28,
                "expected_effect": "stable",
                "critical_values": {
                    "low": 2.8,
                    "high": 6.0
                }
            }
        },
        "success_pattern": {
            "timeline": {
                "days_0_3": "gradual_decrease_bp",
                "days_4_7": "stabilize_therapeutic",
                "days_8_plus": "maintain_therapeutic",
                "labs": "remain_stable"
            }
        },
        "adverse_patterns": [
            {
                "name": "kidney_injury",
                "probability": 0.12,
                "timeline": {
                    "days_0_7": "rising_creatinine",
                    "labs": "creatinine_above_threshold"
                }
            },
            {
                "name": "hyperkalemia",
                "probability": 0.08,
                "timeline": {
                    "days_0_14": "rising_potassium",
                    "labs": "potassium_above_threshold"
                }
            }
        ]
    },

    "Seroquel": {
        "class": "Antipsychotic",
        "monitoring_parameters": {
            "blood_glucose": {
                "baseline_range": (70, 100),
                "therapeutic_range": (70, 140),
                "adverse_threshold_high": 200,
                "monitoring_frequency": "weekly",
                "check_duration": 90,
                "expected_effect": "monitor",
                "critical_values": {
                    "low": 50,
                    "high": 300
                }
            },
            "weight": {
                "baseline": "patient_specific",
                "adverse_threshold": "7_percent_increase",
                "monitoring_frequency": "monthly",
                "check_duration": 90,
                "expected_effect": "monitor"
            },
            "qtc_interval": {
                "baseline_range": (350, 440),
                "therapeutic_range": (350, 470),
                "adverse_threshold_high": 500,
                "monitoring_frequency": "monthly",
                "check_duration": 90,
                "expected_effect": "monitor",
                "critical_values": {
                    "high": 500
                }
            },
            "orthostatic_bp_change": {
                "baseline_range": (0, 20),
                "therapeutic_range": (0, 20),
                "adverse_threshold_high": 30,
                "monitoring_frequency": "weekly",
                "check_duration": 30,
                "expected_effect": "monitor",
                "critical_values": {
                    "high": 40
                }
            }
        },
        "success_pattern": {
            "timeline": {
                "days_0_90": "stable_metabolic",
                "qtc": "remain_therapeutic",
                "orthostatic": "minimal_change"
            }
        },
        "adverse_patterns": [
            {
                "name": "metabolic_syndrome",
                "probability": 0.20,
                "timeline": {
                    "days_0_30": "rising_glucose",
                    "days_30_90": "weight_gain",
                    "labs": "metabolic_changes"
                }
            },
            {
                "name": "qtc_prolongation",
                "probability": 0.10,
                "timeline": {
                    "days_0_30": "qtc_increase",
                    "ecg": "prolonged_qtc"
                }
            }
        ]
    }
}

# Value change patterns for generating data
VALUE_PATTERNS = {
    "gradual_decrease": {
        "description": "Steady decrease within therapeutic goals",
        "day_changes": [-2, -3, -2, -1, -1, 0, -1]  # percentage changes
    },
    "rapid_decrease": {
        "description": "Too rapid decrease indicating adverse effect",
        "day_changes": [-5, -7, -6, -4, -3, -2, -2]
    },
    "minimal_change": {
        "description": "Insufficient therapeutic response",
        "day_changes": [-1, 0, -1, 0, -1, 0, -1]
    },
    "stabilize_therapeutic": {
        "description": "Values stabilize in therapeutic range",
        "day_changes": [-1, 0, 0, 0, -1, 0, 0]
    },
    "maintain_therapeutic": {
        "description": "Maintained in therapeutic range",
        "day_changes": [0, 0, 0, -1, 0, 0, 0]
    },
    "rising_creatinine": {
        "description": "Steady increase in creatinine",
        "day_changes": [5, 8, 10, 12, 15, 18, 20]  # percentage increases
    },
    "rising_potassium": {
        "description": "Steady increase in potassium",
        "day_changes": [2, 3, 4, 5, 6, 7, 8]  # percentage increases
    },
    "rising_glucose": {
        "description": "Steady increase in blood glucose",
        "day_changes": [5, 8, 10, 12, 15, 18, 20]
    }
}