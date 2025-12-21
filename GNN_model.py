import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class AMLGNN(nn.Module):
    def __init__(self, in_dim, edge_attr_dim, hidden_dim=128, heads=4, num_classes=2):
        super().__init__()
        # Encode edge attributes
        self.hidden_dim = hidden_dim
        self.edge_enc = nn.Linear(edge_attr_dim, hidden_dim)

        # Two TransformerConv layers
        self.conv1 = TransformerConv(hidden_dim, hidden_dim // heads,
                                     heads=heads, edge_dim=hidden_dim)
        self.conv2 = TransformerConv(hidden_dim, hidden_dim // heads,
                                     heads=heads, edge_dim=hidden_dim)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.LeakyReLU()
        )

    def forward(self, batch):
        # Ignore batch.x, use a constant embedding for all nodes
        x = torch.ones((batch.n_id.shape[0], self.hidden_dim))
        # Encode edge attributes
        e = self.edge_enc(batch.edge_attr)

        # Message passing
        x = self.conv1(x, batch.edge_index, edge_attr=e)
        x = F.relu(x)
        x = self.conv2(x, batch.edge_index, edge_attr=e)
        x = F.relu(x)

        return self.classifier(x)
