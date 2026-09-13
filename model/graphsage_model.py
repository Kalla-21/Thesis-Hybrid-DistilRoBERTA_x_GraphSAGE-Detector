# graphsage_model.py
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv



class CrossAttentionFusion(nn.Module):

    def __init__(self, text_dim=768, graph_dim=128, attn_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.w_t = nn.Linear(text_dim, attn_dim)
        self.w_g = nn.Linear(graph_dim, attn_dim)
        self.attn_t2g = nn.MultiheadAttention(attn_dim, num_heads=num_heads,
                                               dropout=dropout, batch_first=True)
        self.attn_g2t = nn.MultiheadAttention(attn_dim, num_heads=num_heads,
                                               dropout=dropout, batch_first=True)
        self.norm_t = nn.LayerNorm(attn_dim)
        self.norm_g = nn.LayerNorm(attn_dim)

    def forward(self, h_text, h_graph, x_tab):
        # (N, dim) -> (N, 1, attn_dim); MultiheadAttention wants a sequence dim.
        t = self.w_t(h_text).unsqueeze(1)
        g = self.w_g(h_graph).unsqueeze(1)

        t_attend_g, attn_weights_t2g = self.attn_t2g(query=t, key=g, value=g)
        g_attend_t, attn_weights_g2t = self.attn_g2t(query=g, key=t, value=t)

        t_out = self.norm_t(t_attend_g.squeeze(1) + t.squeeze(1))  # residual + norm
        g_out = self.norm_g(g_attend_t.squeeze(1) + g.squeeze(1))

        fused = torch.cat([t_out, g_out, x_tab], dim=1)

        # Stashed for the attention-visualization XAI phase (C.6) later on --
        # both directions' raw weights, shape (N, num_heads, 1, 1) per call.
        self.last_attn_weights = {"t2g": attn_weights_t2g, "g2t": attn_weights_g2t}
        return fused

    @property
    def output_dim(self):
        return self.w_t.out_features + self.w_g.out_features


class FraudDetectionGraphSAGE(nn.Module):

    def __init__(self, text_dim=768, tabular_dim=3, hidden_channels=256,
                 out_channels=128, attn_dim=128, dropout=0.4,
                 fusion_mode="cross_attention"):
        super().__init__()
        assert fusion_mode in ("concat", "cross_attention"), (
            f"fusion_mode must be 'concat' or 'cross_attention', got {fusion_mode!r}"
        )
        self.fusion_mode = fusion_mode
        self.dropout_rate = dropout

        # The GNN trunk only processes the text embeddings (as in the original design).
        self.conv1 = SAGEConv(text_dim, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)

        if fusion_mode == "concat":
            self.fusion = None
            fused_dim = text_dim + out_channels + tabular_dim
        else:
            self.fusion = CrossAttentionFusion(text_dim=text_dim, graph_dim=out_channels,
                                                attn_dim=attn_dim)
            fused_dim = self.fusion.output_dim + tabular_dim

        # Same MLP classifier head for both fusion conditions, per Ch.3 C.4.
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_text, edge_index, x_tab):
        h = torch.relu(self.bn1(self.conv1(x_text, edge_index)))
        h = torch.dropout(h, p=self.dropout_rate, train=self.training)

        h = torch.relu(self.bn2(self.conv2(h, edge_index)))
        h = torch.dropout(h, p=self.dropout_rate, train=self.training)

        h = torch.relu(self.bn3(self.conv3(h, edge_index)))

        if self.fusion_mode == "concat":
            fused = torch.cat([x_text, h, x_tab], dim=1)
        else:
            fused = self.fusion(x_text, h, x_tab)

        return self.classifier(fused)


class DistilRobertaOnlyClassifier(nn.Module):
    def __init__(self, text_dim=768, hidden=256, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_text):
        return self.net(x_text)


class GraphSAGEOnlyBaseline(nn.Module):

    def __init__(self, num_nodes, rand_dim=768, hidden_channels=256,
                 out_channels=128, dropout=0.4, seed=42):
        super().__init__()
        self.dropout_rate = dropout

        self.node_embed = nn.Embedding(num_nodes, rand_dim)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.node_embed.weight.copy_(torch.randn(num_nodes, rand_dim, generator=g))

        self.conv1 = SAGEConv(rand_dim, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)

        # Same-shaped classifier head as the original single-branch design.
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, edge_index):
        x = self.node_embed.weight
        h = torch.relu(self.bn1(self.conv1(x, edge_index)))
        h = torch.dropout(h, p=self.dropout_rate, train=self.training)

        h = torch.relu(self.bn2(self.conv2(h, edge_index)))
        h = torch.dropout(h, p=self.dropout_rate, train=self.training)

        h = torch.relu(self.bn3(self.conv3(h, edge_index)))
        return self.classifier(h)