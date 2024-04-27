import torch
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, LayerNorm
import torch.nn.functional as F



# GAT Model
# class GAT(torch.nn.Module):
#     def __init__(self, num_features, hidden_channels, num_classes):
#         super(GAT, self).__init__()
#         self.conv1 = GATConv(num_features, hidden_channels)
#         self.conv2 = GATConv(hidden_channels, num_classes)

#     def forward(self, x, edge_index):
#         x = F.elu(self.conv1(x, edge_index))
#         x = self.conv2(x, edge_index)
#         return x

class EnhancedGAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, dropout_rate=0.5):
        super(EnhancedGAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels)
        self.norm1 = LayerNorm(hidden_channels)
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.norm2 = LayerNorm(hidden_channels)
        self.dropout2 = torch.nn.Dropout(dropout_rate)
        
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.norm3 = LayerNorm(hidden_channels)
        self.dropout3 = torch.nn.Dropout(dropout_rate)
        
        self.conv4 = GATConv(hidden_channels, num_classes)
        
        # residual connection
        self.residual = torch.nn.Linear(num_features, num_classes)

    def forward(self, x, edge_index):
        x_input = x  
        x = F.elu(self.conv1(x, edge_index))
        x = self.norm1(x)
        x = self.dropout1(x)
        
        x = F.elu(self.conv2(x, edge_index))
        x = self.norm2(x)
        x = self.dropout2(x)
        
        x = F.elu(self.conv3(x, edge_index))
        x = self.norm3(x)
        x = self.dropout3(x)
        
        intermediate = x  # Select x here as the embedded output of the node

        x = self.conv4(x, edge_index)
        
        # residual connection
        x_res = self.residual(x_input)
        x = x + x_res
        
        return intermediate, x 


