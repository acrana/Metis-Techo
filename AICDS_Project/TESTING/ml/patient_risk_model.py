# patient_risk_model.py
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
import os

class PatientRiskNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Feature dimensions based on your actual data
        self.vital_dim = 6  # heart_rate, bp_sys, bp_dia, resp_rate, o2_sat, temp
        self.lab_dim = 9    # matches your Lab_Ranges table count
        self.condition_embedding_dim = 16
        self.medication_embedding_dim = 16
        
        # Feature processing layers
        self.vitals_encoder = nn.Sequential(
            nn.Linear(self.vital_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.labs_encoder = nn.Sequential(
            nn.Linear(self.lab_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Embedding layers for categorical data
        self.condition_embedding = nn.EmbeddingBag(1000, self.condition_embedding_dim, padding_idx=0)
        self.medication_embedding = nn.EmbeddingBag(1000, self.medication_embedding_dim, padding_idx=0)
        
        # New layer for medication-specific features
        self.med_specific_layer = nn.Linear(1, 16)
        
        # Risk prediction layers
        combined_dim = 32 + 32 + self.condition_embedding_dim + self.medication_embedding_dim + 16  # Added 16 for med_specific
        self.risk_predictor = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # [ADE risk, Interaction risk, Overall risk]
        )

    def forward(self, vitals, labs, conditions, medications, med_specific=None):
        # Process vitals
        vital_features = self.vitals_encoder(vitals)
        
        # Process labs
        lab_features = self.labs_encoder(labs)
        
        # Process conditions and medications using EmbeddingBag
        condition_features = self.condition_embedding(conditions.long())
        medication_features = self.medication_embedding(medications.long())
        
        # Process medication-specific features if available
        if med_specific is not None:
            med_specific_features = self.med_specific_layer(med_specific)
            # Combine all features
            combined = torch.cat([
                vital_features,
                lab_features,
                condition_features,
                medication_features,
                med_specific_features
            ], dim=1)
        else:
            # Combine features without med_specific
            combined = torch.cat([
                vital_features,
                lab_features,
                condition_features,
                medication_features,
                torch.zeros(vitals.size(0), 16).to(vitals.device)  # Zero padding for consistency
            ], dim=1)
        
        # Generate risk predictions
        risks = self.risk_predictor(combined)
        return torch.sigmoid(risks)