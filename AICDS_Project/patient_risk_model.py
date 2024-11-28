import torch
import torch.nn as nn

class PatientRiskNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.vital_dim = 6
        self.lab_dim = 9
        self.condition_embedding_dim = 16
        self.medication_embedding_dim = 16
        self.med_specific_dim = 6
        
        # Fixed vitals encoder for single samples
        self.vitals_encoder = nn.Sequential(
            nn.Linear(self.vital_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Linear(24, 16),
            nn.ReLU()
        )

        self.vital_attention = nn.Sequential(
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Softmax(dim=1)
        )
        
        # Rest of the model remains same
        self.labs_encoder = nn.Sequential(
            nn.Linear(self.lab_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        self.condition_embedding = nn.EmbeddingBag(1000, self.condition_embedding_dim, padding_idx=0)
        self.medication_embedding = nn.EmbeddingBag(1000, self.medication_embedding_dim, padding_idx=0)
        
        self.med_specific_encoder = nn.Sequential(
            nn.Linear(self.med_specific_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16)
        )
        
        combined_dim = 16 + 16 + self.condition_embedding_dim + self.medication_embedding_dim + 16
        
        self.risk_predictor = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3)
        )

    def normalize_vitals(self, vitals):
        ranges = {
            'heart_rate': (60, 100),
            'bp_systolic': (90, 140),
            'bp_diastolic': (60, 90),
            'resp_rate': (12, 20),
            'o2_sat': (95, 100),
            'temp': (36.5, 37.5)
        }
        
        normalized = torch.zeros_like(vitals)
        for i, (min_val, max_val) in enumerate(ranges.values()):
            normalized[:, i] = (vitals[:, i] - min_val) / (max_val - min_val)
            normalized[:, i] = torch.clamp(normalized[:, i], -2, 2)
        
        return normalized

    def forward(self, vitals, labs, conditions, medications, med_specific=None):
        # Process normalized vitals with attention
        normalized_vitals = self.normalize_vitals(vitals)
        vital_features = self.vitals_encoder(normalized_vitals)
        attention_weights = self.vital_attention(vital_features)
        vital_features = vital_features * attention_weights
        
        lab_features = self.labs_encoder(labs)
        condition_features = self.condition_embedding(conditions.long())
        medication_features = self.medication_embedding(medications.long())
        
        if med_specific is not None:
            med_specific_features = self.med_specific_encoder(med_specific)
        else:
            med_specific_features = torch.zeros(vitals.size(0), 16).to(vitals.device)
        
        combined = torch.cat([
            vital_features,
            lab_features,
            condition_features,
            medication_features,
            med_specific_features
        ], dim=1)
        
        risks = self.risk_predictor(combined)
        return torch.sigmoid(risks)