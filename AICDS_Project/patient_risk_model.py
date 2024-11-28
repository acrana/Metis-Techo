import torch
import torch.nn as nn

class PatientRiskNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Feature dimensions
        self.vital_dim = 6  # heart_rate, bp_sys, bp_dia, resp_rate, o2_sat, temp
        self.lab_dim = 9    # From your Lab_Ranges table
        self.condition_embedding_dim = 16
        self.medication_embedding_dim = 16
        self.med_specific_dim = 6  # From your med_specific features
        
        # Feature extraction layers
        self.vitals_encoder = nn.Sequential(
            nn.Linear(self.vital_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        self.labs_encoder = nn.Sequential(
            nn.Linear(self.lab_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Categorical data embeddings
        self.condition_embedding = nn.EmbeddingBag(1000, self.condition_embedding_dim, padding_idx=0)
        self.medication_embedding = nn.EmbeddingBag(1000, self.medication_embedding_dim, padding_idx=0)
        
        # Medication-specific feature processing
        self.med_specific_encoder = nn.Sequential(
            nn.Linear(self.med_specific_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16)
        )
        
        # Combine all features
        combined_dim = 16 + 16 + self.condition_embedding_dim + self.medication_embedding_dim + 16
        
        # Final risk prediction layers
        self.risk_predictor = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3)
        )

    def forward(self, vitals, labs, conditions, medications, med_specific=None):
        # Process each feature type
        vital_features = self.vitals_encoder(vitals)
        lab_features = self.labs_encoder(labs)
        condition_features = self.condition_embedding(conditions.long())
        medication_features = self.medication_embedding(medications.long())
        
        # Process med_specific features if available
        if med_specific is not None:
            med_specific_features = self.med_specific_encoder(med_specific)
        else:
            med_specific_features = torch.zeros(vitals.size(0), 16).to(vitals.device)
        
        # Combine all features
        combined = torch.cat([
            vital_features,
            lab_features,
            condition_features,
            medication_features,
            med_specific_features
        ], dim=1)
        
        # Generate risk predictions
        risks = self.risk_predictor(combined)
        return torch.sigmoid(risks)