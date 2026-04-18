import os
import json
from xml.parsers.expat import model
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import f1_score, recall_score, precision_score
from utils import load_and_split_data, plot_training_history, DEVICE, GNN_DEVICE, JobTextDataset
from models.distilroberta_encoder import DistilRobertaModule
from models.graphsage_model import FraudDetectionGraphSAGE


# ------------------------------------------------------------------ #
#  Hyperparameters 
# ------------------------------------------------------------------ #
HP = {
    "lr":              0.00075,   # was 0.01  too high, caused epoch-85 spike
    "epochs":          200,     # more runway; early stopping will cut it short
    "patience":        25,      # early stopping patience (epochs without val F1 improvement)
    "grad_clip":       1.0,     # max gradient norm — prevents loss spikes
    "weight_caps":     [4.0, 6.0, 8.0, 10.0],  # pos_weight caps to sweep during training
    "pos_weight_cap":  10.0,     # cap raw class ratio; 19.65 caused precision collapse
    "lr_patience":     10,      # Reduce Plateau patience
    "lr_factor":       0.5,     # LR decay factor on plateau
    "threshold_steps": 50,      # number of thresholds to sweep during val tuning
}


def tune_threshold(model, graph_data):
    model.eval()
    with torch.no_grad():
        # ADDED graph_data.x_tab HERE
        logits = model(graph_data.x, graph_data.edge_index, graph_data.x_tab)
        probs  = torch.sigmoid(logits).squeeze()

    y_true = graph_data.y[graph_data.val_mask].cpu().numpy()
    probs  = probs[graph_data.val_mask].cpu().numpy()

    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.linspace(0.1, 0.9, HP["threshold_steps"]):
        preds = (probs >= thresh).astype(float)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1     = score
            best_thresh = thresh

    print(f"[INFO] Best threshold: {best_thresh:.3f}  (Val F1 = {best_f1:.4f})")
    return float(best_thresh)


class FocalLoss(torch.nn.Module):
     def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

     def forward(self, logits, targets):
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight
        )
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss
        return focal_loss.mean()

def train():
    print("========== STARTING TRAINING SCRIPT ==========")
    os.makedirs("outputs", exist_ok=True)

    # --- STAGE 1: Data Ingestion ---
    # Catch the 6th returned variable: tabular_features
    df, edges_arr, train_mask, val_mask, test_mask, tabular_features = load_and_split_data()

    num_licit = (df.iloc[train_mask.numpy()]["label"] == 0).sum()
    num_scam  = (df.iloc[train_mask.numpy()]["label"] == 1).sum()
    raw_weight = num_licit / max(num_scam, 1)
    print(f"[INFO] Raw mathematical class weight calculated at: {raw_weight:.2f}")

    # --- STAGE 2: Text Embedding Extraction ---
    emb_path = "outputs/embeddings.npy"
    if os.path.exists(emb_path):
        print("\n--- STAGE 2: TEXT EMBEDDING EXTRACTION (CACHED) ---")
        embeddings = np.load(emb_path)
    else:
        print("\n--- STAGE 2: TEXT EMBEDDING EXTRACTION ---")
        text_module = DistilRobertaModule(DEVICE)
        full_dataset = JobTextDataset(df["full_text"].values, df["label"].values, text_module.tokenizer)
        full_loader  = DataLoader(full_dataset, batch_size=16, shuffle=False)
        embeddings = text_module.extract_embeddings(full_loader)
        np.save(emb_path, embeddings)

    # --- STAGE 3: Graph Construction ---
    print("\n--- STAGE 3: GRAPH CONSTRUCTION ---")
    
    print("[PROCESS] Preparing separated text and tabular features for Late Fusion...")
    x_text     = torch.tensor(embeddings, dtype=torch.float)
    x_tab      = torch.tensor(tabular_features, dtype=torch.float)
    y          = torch.tensor(df["label"].values, dtype=torch.float)
    edge_index = torch.tensor(edges_arr, dtype=torch.long)

    # PyG Data objects can hold custom attributes perfectly
    graph_data            = Data(x=x_text, edge_index=edge_index, y=y)
    graph_data.x_tab      = x_tab  
    graph_data.train_mask = train_mask
    graph_data.val_mask   = val_mask
    graph_data.test_mask  = test_mask

    print("[PROCESS] Saving raw graph data to disk for testing phase...")
    torch.save(graph_data, "outputs/graph_data.pt")
 
    
    # --- STAGE 4: GRAPHSAGE TRAINING (WEIGHT SWEEP) ---
    print("\n--- STAGE 4: GRAPHSAGE TRAINING ---")
    graph_data = graph_data.to(GNN_DEVICE)

    best_global_f1     = 0.0
    best_global_weight = None
    best_history       = None

    for cap in HP["weight_caps"]:
        print(f"\n[PROCESS] --- Testing pos_weight capped at {cap:.1f} ---")
        capped_weight = min(raw_weight, cap)
        pos_weight    = torch.tensor([capped_weight], dtype=torch.float).to(GNN_DEVICE)

        # UPDATE: Pass both the text dimensions and tabular dimensions
        model = FraudDetectionGraphSAGE(
            in_channels=graph_data.x.shape[1], 
            tabular_dim=graph_data.x_tab.shape[1]
        ).to(GNN_DEVICE)
        
        criterion = FocalLoss(gamma=2.0, pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=HP["lr"], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=HP["lr_factor"], patience=HP["lr_patience"]
        )

        history = {"loss": [], "precision": [], "recall": [], "f1": []}
        best_run_f1, best_run_epoch, patience_count = 0.0, 0, 0

        for epoch in range(HP["epochs"]):
            model.train()
            optimizer.zero_grad()
            
            # UPDATE: Pass x_tab into the forward pass
            out = model(graph_data.x, graph_data.edge_index, graph_data.x_tab)
            loss = criterion(
                out[graph_data.train_mask].squeeze(),
                graph_data.y[graph_data.train_mask]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), HP["grad_clip"])
            optimizer.step()

            model.eval()
            with torch.no_grad():
                # UPDATE: Pass x_tab into the validation pass
                logits = model(graph_data.x, graph_data.edge_index, graph_data.x_tab)
                probs  = torch.sigmoid(logits).squeeze()
                preds  = (probs > 0.5).float()

                y_true = graph_data.y[graph_data.val_mask].cpu().numpy()
                y_pred = preds[graph_data.val_mask].cpu().numpy()

                val_f1        = f1_score(y_true, y_pred, zero_division=0)
                val_recall    = recall_score(y_true, y_pred, zero_division=0)
                val_precision = precision_score(y_true, y_pred, zero_division=0)

            history["loss"].append(loss.item())
            history["f1"].append(val_f1)
            history["recall"].append(val_recall)
            history["precision"].append(val_precision)

            scheduler.step(val_f1)

            if val_f1 > best_run_f1:
                best_run_f1    = val_f1
                best_run_epoch = epoch + 1
                patience_count = 0

                if val_f1 > best_global_f1:
                    best_global_f1     = val_f1
                    best_global_weight = cap
                    best_history       = {k: list(v) for k, v in history.items()}
                    torch.save(model.state_dict(), "outputs/graphsage_weights_best.pth")
            else:
                patience_count += 1

            if (epoch + 1) % 10 == 0:
                print(
                    f"   -> Epoch {epoch+1:03d}/{HP['epochs']} | "
                    f"Loss: {loss.item():.4f} | Val F1: {val_f1:.4f} | "
                    f"P: {val_precision:.4f} | R: {val_recall:.4f}"
                )

            if patience_count >= HP["patience"]:
                print(f"   [INFO] Early stopping at epoch {epoch+1}. Best Val F1 = {best_run_f1:.4f} @ epoch {best_run_epoch}")
                break
            
    print(f"\n[INFO] Sweep Complete! Global Best Val F1 = {best_global_f1:.4f} (Found with weight cap {best_global_weight})")

    print("\n[PROCESS] Loading global best checkpoint for threshold tuning...")
    model = FraudDetectionGraphSAGE(in_channels=graph_data.x.shape[1]).to(GNN_DEVICE)
    model.load_state_dict(torch.load("outputs/graphsage_weights_best.pth", map_location=GNN_DEVICE, weights_only=True))

    best_threshold = tune_threshold(model, graph_data)
    with open("outputs/best_threshold.json", "w") as f:
        json.dump({"threshold": best_threshold}, f)

    plot_training_history(best_history, f"Hybrid_GraphSAGE_Cap_{best_global_weight}")

    # Re-load the best checkpoint and save it under the name test.py expects
    model.load_state_dict(torch.load("outputs/graphsage_weights_best.pth", map_location=GNN_DEVICE, weights_only=True))
    torch.save(model.state_dict(), "outputs/graphsage_weights.pth")
    print("[INFO] Best weights saved to outputs/graphsage_weights.pth")
    print("========== TRAINING SCRIPT FINISHED ==========\n")

if __name__ == "__main__":
    train()