#train.py
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from losses import FocalLoss
from sklearn.metrics import (
    f1_score, recall_score, precision_score, accuracy_score, matthews_corrcoef
)

from utils import (
    DEVICE, GNN_DEVICE, JobTextDataset, set_seed,
    load_and_split_data, stratified_kfold_masks, 
)

from models.distilroberta_encoder import DistilRobertaModule
from models.graphsage_model import (
    FraudDetectionGraphSAGE, DistilRobertaOnlyClassifier, GraphSAGEOnlyBaseline,
)

CV_CONFIG = {
    "k_folds":         5,
    "seed":            42,
    "pos_weight_cap":  4.0,   #<-- possible change
    "lr":              0.00075,   #<-- possible change
    "epochs":          200,
    "patience":        25,
    "grad_clip":       1.0,
    "lr_patience":     10,
    "lr_factor":       0.5,
    "threshold_steps": 50,
}
CONDITIONS = ["distilroberta_only", "graphsage_only", "late_concat", "cross_attention"]

def tune_threshold(probs, labels, steps=50):
    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.linspace(0.1, 0.9, steps):
        preds = (probs >= thresh).astype(float)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1:
            best_f1, best_thresh = score, thresh
    return float(best_thresh), float(best_f1)


def get_metrics(y_true, y_pred):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "mcc":       matthews_corrcoef(y_true, y_pred),
    }


def train_one_model(model, forward_fn, y, train_mask, val_mask, test_mask,
                     pos_weight, cfg, device):

    model = model.to(device)
    criterion = FocalLoss(gamma=2.0, pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg["lr_factor"], patience=cfg["lr_patience"]
    )

    best_val_f1, best_state, patience_count = 0.0, None, 0

    for epoch in range(cfg["epochs"]):
        model.train()
        optimizer.zero_grad()
        logits = forward_fn(model)
        loss = criterion(logits[train_mask].squeeze(), y[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = forward_fn(model)
            probs = torch.sigmoid(logits).squeeze()

        # threshold-independent val F1, used for early stopping 
        val_probs_np = probs[val_mask].cpu().numpy()
        val_y_np = y[val_mask].cpu().numpy()
        _, val_f1 = tune_threshold(val_probs_np, val_y_np, cfg["threshold_steps"])

        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= cfg["patience"]:
            break

    if best_state is None:
        print("[WARN] val_f1 never improved above 0.0 within patience window — "
              "falling back to the final epoch's weights instead of crashing.")
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # final evaluation, using the best checkpoint loaded back in
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = forward_fn(model)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    y_np = y.cpu().numpy()
    val_mask_np = val_mask.cpu().numpy()
    test_mask_np = test_mask.cpu().numpy()

    best_thresh, _ = tune_threshold(probs[val_mask_np], y_np[val_mask_np], cfg["threshold_steps"])

    test_preds = (probs[test_mask_np] >= best_thresh).astype(float)
    metrics = get_metrics(y_np[test_mask_np], test_preds)
    return metrics, best_thresh, probs


def run():
    set_seed(CV_CONFIG["seed"])
    os.makedirs("outputs", exist_ok=True)

    df, edges_arr, _, _, _, tabular_features = load_and_split_data()
    labels_arr = df["label"].values

    emb_path = "outputs/embeddings.npy"
    if os.path.exists(emb_path):
        print("[INFO] Loading cached text embeddings...")
        embeddings = np.load(emb_path)
    else:
        text_module = DistilRobertaModule(DEVICE)
        full_dataset = JobTextDataset(df["full_text"].values, df["label"].values, text_module.tokenizer)
        full_loader = DataLoader(full_dataset, batch_size=16, shuffle=False)
        embeddings = text_module.extract_embeddings(full_loader)
        np.save(emb_path, embeddings)

    x_text = torch.tensor(embeddings, dtype=torch.float).to(GNN_DEVICE)
    x_tab = torch.tensor(tabular_features, dtype=torch.float).to(GNN_DEVICE)
    y = torch.tensor(labels_arr, dtype=torch.float).to(GNN_DEVICE)
    edge_index = torch.tensor(edges_arr, dtype=torch.long).to(GNN_DEVICE)
    num_nodes = x_text.shape[0]

    results = {cond: [] for cond in CONDITIONS}

    for fold_i, train_mask, val_mask, test_mask in stratified_kfold_masks(
        labels_arr, k=CV_CONFIG["k_folds"], seed=CV_CONFIG["seed"]
    ):
        print(f"\n===== FOLD {fold_i + 1}/{CV_CONFIG['k_folds']} =====")
        train_mask = train_mask.to(GNN_DEVICE)
        val_mask = val_mask.to(GNN_DEVICE)
        test_mask = test_mask.to(GNN_DEVICE)

        num_licit = (y[train_mask] == 0).sum().item()
        num_scam = (y[train_mask] == 1).sum().item()
        raw_weight = num_licit / max(num_scam, 1)
        capped_weight = min(raw_weight, CV_CONFIG["pos_weight_cap"])
        pos_weight = torch.tensor([capped_weight], dtype=torch.float).to(GNN_DEVICE)

        # standalone DistilRoBERTa 
        set_seed(CV_CONFIG["seed"] + fold_i)
        model = DistilRobertaOnlyClassifier(text_dim=x_text.shape[1])
        metrics, thresh, _probs = train_one_model(
            model, lambda m: m(x_text), y, train_mask, val_mask, test_mask,
            pos_weight, CV_CONFIG, GNN_DEVICE,
        )
        print(f"[distilroberta_only] F1={metrics['f1']:.4f} MCC={metrics['mcc']:.4f} (thresh={thresh:.3f})")
        results["distilroberta_only"].append(metrics)

        # standalone GraphSAGE only 
        set_seed(CV_CONFIG["seed"] + fold_i)
        model = GraphSAGEOnlyBaseline(
            num_nodes=num_nodes, rand_dim=x_text.shape[1], seed=CV_CONFIG["seed"]
        )
        metrics, thresh, _probs = train_one_model(
            model, lambda m: m(edge_index), y, train_mask, val_mask, test_mask,
            pos_weight, CV_CONFIG, GNN_DEVICE,
        )
        print(f"[graphsage_only]     F1={metrics['f1']:.4f} MCC={metrics['mcc']:.4f} (thresh={thresh:.3f})")
        results["graphsage_only"].append(metrics)

        # late-fusion concatenation (ablation baseline)
        set_seed(CV_CONFIG["seed"] + fold_i)
        model = FraudDetectionGraphSAGE(
            text_dim=x_text.shape[1], tabular_dim=x_tab.shape[1], fusion_mode="concat"
        )
        metrics, thresh, _probs = train_one_model(
            model, lambda m: m(x_text, edge_index, x_tab), y, train_mask, val_mask, test_mask,
            pos_weight, CV_CONFIG, GNN_DEVICE,
        )
        print(f"[late_concat]        F1={metrics['f1']:.4f} MCC={metrics['mcc']:.4f} (thresh={thresh:.3f})")
        results["late_concat"].append(metrics)

        # cross-attention fusion (primary proposed architecture) 
        set_seed(CV_CONFIG["seed"] + fold_i)
        model = FraudDetectionGraphSAGE(
            text_dim=x_text.shape[1], tabular_dim=x_tab.shape[1], fusion_mode="cross_attention"
        )
        metrics, thresh, _probs = train_one_model(
            model, lambda m: m(x_text, edge_index, x_tab), y, train_mask, val_mask, test_mask,
            pos_weight, CV_CONFIG, GNN_DEVICE,
        )
        print(f"[cross_attention]    F1={metrics['f1']:.4f} MCC={metrics['mcc']:.4f} (thresh={thresh:.3f})")
        results["cross_attention"].append(metrics)

    with open("outputs/cv_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[INFO] Saved per-fold results to outputs/cv_results.json")

    print("\n===== SUMMARY (mean ± std across folds) =====")
    for cond in CONDITIONS:
        f1s = [r["f1"] for r in results[cond]]
        mccs = [r["mcc"] for r in results[cond]]
        print(f"{cond:20s} F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}  "
              f"MCC={np.mean(mccs):.4f}±{np.std(mccs):.4f}")


if __name__ == "__main__":
    run()