# graphsage_model.py
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class CrossAttentionFusion(nn.Module):
    # main fusion module that combines the text and graph representations using cross-attention. 
    # It takes the text embeddings, graph embeddings from multiple hops, and tabular features as input, 
    # and outputs a fused representation that can be used for classification.
    
    def __init__(self, text_dim=768, graph_dims=(256, 256, 128), attn_dim=128, # Linear 768 to 128, shrinks the text vector into the attention space.
                 num_heads=4, dropout=0.1):  # three Linear layers (256 to 128, 256 to 128, 128 to 128), one per graph layer output.
        super().__init__() 
        self.text_dim = text_dim                 
        self.graph_out = graph_dims[-1]          
        self.attn_dim = attn_dim
        self.w_t = nn.Linear(text_dim, attn_dim)
        self.w_g = nn.ModuleList([nn.Linear(d, attn_dim) for d in graph_dims])

        self.attn_t2g = nn.MultiheadAttention(attn_dim, num_heads=num_heads, # multi-head attention where the text asks and the graph views answer.
                                              dropout=dropout, batch_first=True)
        self.attn_pool = nn.MultiheadAttention(attn_dim, num_heads=num_heads, # a second multi-head attention used to pool 
                                                                              # the three graph views into one vector.
                                               dropout=dropout, batch_first=True)

        self.norm_t = nn.LayerNorm(attn_dim)
        self.norm_g = nn.LayerNorm(attn_dim)

        # learned query used to pool the 3-length graph sequence into 1 vector
        self.pool_query = nn.Parameter(torch.randn(1, 1, attn_dim))

    def forward(self, h_text, graph_hops, x_tab):
        N = h_text.shape[0]

        # 1. project to attention space
        t = self.w_t(h_text).unsqueeze(1) # reshaped to (N, 1, 128): one text token per posting
        g_seq = torch.stack([proj(h) for proj, h in zip(self.w_g, graph_hops)], dim=1)

        # 2. attention
        t_attend_g, attn_weights_t2g = self.attn_t2g(query=t, key=g_seq, value=g_seq)
        t_out = self.norm_t(t_attend_g.squeeze(1) + t.squeeze(1))

        pool_q = self.pool_query.expand(N, -1, -1) # a three-token sequence per posting (1-step, 2-step and 3-step neighbourhood summaries)
        g_pooled, attn_weights_pool = self.attn_pool(query=pool_q, key=g_seq, value=g_seq)
        g_out = self.norm_g(g_pooled.squeeze(1)) # the attended result added back to the projected text, then normalized

        # 3. keep the raw text and final-hop vectors next to the attended ones
        fused = torch.cat([h_text, graph_hops[-1], t_out, g_out, x_tab], dim=1)

        self.last_attn_weights = {"t2g": attn_weights_t2g, "graph_pool": attn_weights_pool}
        return fused  

    @property
    def output_dim(self):
        # raw text + final-hop graph + attended text + pooled graph (tabular is added by the caller)
        return self.text_dim + self.graph_out + 2 * self.attn_dim   
        # The raw text vector and the final graph vector are kept next to the attended ones, 
        # so this block contains everything late fusion has, plus the attention features.


class FraudDetectionGraphSAGE(nn.Module):

    def __init__(self, text_dim=768, tabular_dim=11, hidden_channels=256,  
                 out_channels=128, attn_dim=128, dropout=0.4,
                 fusion_mode="cross_attention"):
        super().__init__() # asserts that the fusion_mode is either "concat" or "cross_attention", 
        # and initializes the GraphSAGE model with the specified dimensions and parameters.
        assert fusion_mode in ("concat", "cross_attention"), (
            f"fusion_mode must be 'concat' or 'cross_attention', got {fusion_mode!r}" 
        )
        self.fusion_mode = fusion_mode
        self.dropout_rate = dropout

        # GraphSAGE layers: three SAGEConv layers with batch normalization and ReLU activation.
        self.conv1 = SAGEConv(text_dim, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)

        # Fusion layer: either concatenation or cross-attention fusion, depending on the specified fusion_mode.
        if fusion_mode == "concat":
            self.fusion = None
            fused_dim = text_dim + out_channels + tabular_dim
        else: # cross_attention: layer that combines the text and graph representations using multi-head attention, 
            #followed by a pooling operation to aggregate the graph representations.
            self.fusion = CrossAttentionFusion(
                text_dim=text_dim,
                graph_dims=(hidden_channels, hidden_channels, out_channels),
                attn_dim=attn_dim,
            )
            fused_dim = self.fusion.output_dim + tabular_dim

        # same MLP head for both fusion conditions
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
        # forward pass of the model, which takes the text embeddings, 
        # graph edge indices, and tabular features as input,and outputs the predicted logits for each posting. 
        # The model first applies the GraphSAGE layers to the text embeddings and graph structure, 
        # then fuses the resulting representations with the tabular features,
        # and finally passes the fused representation through the MLP classifier to obtain the final logits.
        h1 = torch.relu(self.bn1(self.conv1(x_text, edge_index)))
        h1_drop = torch.dropout(h1, p=self.dropout_rate, train=self.training)

        h2 = torch.relu(self.bn2(self.conv2(h1_drop, edge_index)))
        h2_drop = torch.dropout(h2, p=self.dropout_rate, train=self.training)

        h3 = torch.relu(self.bn3(self.conv3(h2_drop, edge_index)))

        if self.fusion_mode == "concat":
            fused = torch.cat([x_text, h3, x_tab], dim=1)
        else:
            fused = self.fusion(x_text, [h1, h2, h3], x_tab)

        return self.classifier(fused)


class DistilRobertaOnlyClassifier(nn.Module):
    # a simple MLP classifier that takes the DistilRoBERTa embeddings as input and outputs the predicted logits for each posting.
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


class TextTabularMLP(nn.Module):
    # a simple MLP classifier that takes both text and tabular features as input and outputs the predicted logits for each posting.
    def __init__(self, text_dim, tab_dim, hidden=256, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(text_dim + tab_dim, hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x_text, x_tab):
        return self.net(torch.cat([x_text, x_tab], dim=1))


class GraphSAGEOnlyBaseline(nn.Module):
    # a simple GraphSAGE model that takes the text embeddings and graph structure as input and outputs the predicted logits for each posting.
    # It does not use any tabular features or fusion mechanisms, and serves as a baseline for evaluating the performance of the full model.
    def __init__(self, num_nodes, rand_dim=768, hidden_channels=256,
                 out_channels=128, dropout=0.4, seed=42):
        super().__init__()
        self.dropout_rate = dropout

        self.node_embed = nn.Embedding(num_nodes, rand_dim)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.node_embed.weight.copy_(torch.randn(num_nodes, rand_dim, generator=g))

        # GraphSAGE layers: three SAGEConv layers with batch normalization and ReLU activation.
        self.conv1 = SAGEConv(rand_dim, hidden_channels) # 768 -> 256
        self.conv2 = SAGEConv(hidden_channels, hidden_channels) # 256 -> 256
        self.conv3 = SAGEConv(hidden_channels, out_channels) # 256 -> 128

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)

        # MLP head: a simple feedforward network that takes the final graph representation 
        # as input and outputs the predicted logits for each posting.
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
        # forward pass of the model, which takes the graph edge indices as input and outputs the predicted logits for each posting.
        # The model first applies the GraphSAGE layers to the node embeddings and graph structure,
        x = self.node_embed.weight
        h = torch.relu(self.bn1(self.conv1(x, edge_index)))
        h = torch.dropout(h, p=self.dropout_rate, train=self.training)

        h = torch.relu(self.bn2(self.conv2(h, edge_index)))
        h = torch.dropout(h, p=self.dropout_rate, train=self.training)

        h = torch.relu(self.bn3(self.conv3(h, edge_index)))
        return self.classifier(h)