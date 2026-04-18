import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class FraudDetectionGraphSAGE(nn.Module):
    # Added tabular_dim=3 to the initialization
    def __init__(self, in_channels, tabular_dim=3, hidden_channels=256, out_channels=128, dropout=0.4):
        super().__init__()
        self.dropout_rate = dropout

        # The GNN only processes the text embeddings
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)

        # The MLP Classifier now takes the GNN output PLUS the raw tabular features
        self.classifier = nn.Sequential(
            nn.Linear(out_channels + tabular_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    # Forward pass now expects text features (x_text) and tabular features (x_tab) separately
    def forward(self, x_text, edge_index, x_tab):
        # 1. Pass text through the GNN
        h = torch.relu(self.bn1(self.conv1(x_text, edge_index)))
        h = torch.dropout(h, p=self.dropout_rate, train=self.training)
        
        h = torch.relu(self.bn2(self.conv2(h, edge_index)))
        h = torch.dropout(h, p=self.dropout_rate, train=self.training)
        
        h = torch.relu(self.bn3(self.conv3(h, edge_index)))
        
        # 2. LATE FUSION: Concatenate the graph-aware text with the hard tabular facts
        fused_h = torch.cat([h, x_tab], dim=1)
        
        # 3. Final Classification
        return self.classifier(fused_h)