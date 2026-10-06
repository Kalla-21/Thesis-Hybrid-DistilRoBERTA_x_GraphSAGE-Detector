# train_cv.py
import os
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, fbeta_score, recall_score, precision_score,
    accuracy_score, matthews_corrcoef, average_precision_score,
)

from losses import FocalLoss
from utils import (
    DEVICE, GNN_DEVICE, JobTextDataset, set_seed,
    load_and_split_data, stratified_kfold_masks, connectivity_breakdown,
)
from models.distilroberta_encoder import DistilRobertaModule
from models.graphsage_model import (
    FraudDetectionGraphSAGE, DistilRobertaOnlyClassifier,
    GraphSAGEOnlyBaseline, TextTabularMLP,
)

CV_CONFIG = {
    "k_folds":         5,
    "seed":            42,
    "split":           "random",   
    "n_seeds":         1,
    "pos_weight_cap":  3.5,
    "lr":              0.00075,
    "epochs":          200,
    "patience":        25,
    "grad_clip":       1.0,
    "lr_patience":     10,
    "lr_factor":       0.5,
    "threshold_steps": 80,
    "threshold_min":   0.02,
    "beta":            1.0,
    "gamma":           2.0,
    "tab_mode":        "paper",
    "edge_types":      ("company", "title"),
    "cap_cliques":     False,
    "tag": "",
}
CONDITIONS = ["distilroberta_only", "text_tabular", "graphsage_only",
              "late_concat", "cross_attention"]


def tune_threshold(probs, labels, steps=80, beta=2.0, t_min=0.02):
    grid = np.linspace(t_min, 0.9, steps)
    scores = np.array([
        fbeta_score(labels, (probs >= t).astype(float), beta=beta, zero_division=0)
        for t in grid
    ])
    smooth = np.convolve(scores, np.ones(5) / 5, mode="same")
    i = int(smooth.argmax())
    return float(grid[i]), float(scores[i])


def get_metrics(y_true, y_pred, probs=None):
    m = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "f2":        fbeta_score(y_true, y_pred, beta=2.0, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "mcc":       matthews_corrcoef(y_true, y_pred),
    }
    if probs is not None:
        m["pr_auc"] = average_precision_score(y_true, probs)
    return m


def train_one_model(model, forward_fn, y, train_mask, val_mask, test_mask,
                    pos_weight, cfg, device):
    model = model.to(device)
    criterion = FocalLoss(gamma=cfg["gamma"], pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg["lr_factor"], patience=cfg["lr_patience"]
    )

    best_val, best_state, patience_count = 0.0, None, 0

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
            probs = torch.sigmoid(forward_fn(model)).squeeze()

        # val PR-AUC: threshold-free, used for early stopping and LR scheduling
        val_score = average_precision_score(
            y[val_mask].cpu().numpy(), probs[val_mask].cpu().numpy()
        )
        scheduler.step(val_score)

        if val_score > best_val:
            best_val = val_score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= cfg["patience"]:
            break

    if best_state is None:
        print("[WARN] val score never improved; using final epoch weights.")
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(forward_fn(model)).squeeze().cpu().numpy()

    y_np = y.cpu().numpy()
    val_np = val_mask.cpu().numpy()
    test_np = test_mask.cpu().numpy()

    best_thresh, _ = tune_threshold(
        probs[val_np], y_np[val_np],
        cfg["threshold_steps"], cfg["beta"], cfg["threshold_min"],
    )
    test_probs = probs[test_np]
    test_preds = (test_probs >= best_thresh).astype(float)
    metrics = get_metrics(y_np[test_np], test_preds, test_probs)
    return metrics, best_thresh, probs


def get_embeddings(df, col, max_len, path):
    if os.path.exists(path):
        e = np.load(path)
        assert len(e) == len(df), f"{path} doesn't match the dataset; delete it."
        return e
    tm = DistilRobertaModule(DEVICE)
    ds = JobTextDataset(df[col].values, df["label"].values, tm.tokenizer, max_len=max_len)
    e = tm.extract_embeddings(DataLoader(ds, batch_size=16, shuffle=False))
    np.save(path, e)
    return e


def run():
    set_seed(CV_CONFIG["seed"])
    os.makedirs("outputs", exist_ok=True)

    df, edges_arr, _, _, _, tabular_features = load_and_split_data()
    labels_arr = df["label"].values

    emb_job = get_embeddings(df, "job_text", 512, "outputs/emb_job_512.npy")
    emb_co = get_embeddings(df, "company_profile", 256, "outputs/emb_company_256.npy")
    embeddings = get_embeddings(df, "full_text", 512, "outputs/emb_full_512.npy")

    x_text = torch.tensor(embeddings, dtype=torch.float).to(GNN_DEVICE)
    x_tab = torch.tensor(tabular_features, dtype=torch.float).to(GNN_DEVICE)
    y = torch.tensor(labels_arr, dtype=torch.float).to(GNN_DEVICE)
    edge_index = torch.tensor(edges_arr, dtype=torch.long).to(GNN_DEVICE)
    num_nodes = x_text.shape[0]
    td, tt = x_text.shape[1], x_tab.shape[1]

    results = {cond: [] for cond in CONDITIONS}
    groups = df["group_id"].values if CV_CONFIG["split"] == "group" else None

    for fold_i, train_mask, val_mask, test_mask in stratified_kfold_masks(
        labels_arr, k=CV_CONFIG["k_folds"], seed=CV_CONFIG["seed"], groups=groups
    ):
        print(f"\n===== FOLD {fold_i + 1}/{CV_CONFIG['k_folds']} ({CV_CONFIG['split']} split) =====")
        train_mask = train_mask.to(GNN_DEVICE)
        val_mask = val_mask.to(GNN_DEVICE)
        test_mask = test_mask.to(GNN_DEVICE)

        num_licit = (y[train_mask] == 0).sum().item()
        num_scam = (y[train_mask] == 1).sum().item()
        capped = min(num_licit / max(num_scam, 1), CV_CONFIG["pos_weight_cap"])
        pos_weight = torch.tensor([capped], dtype=torch.float).to(GNN_DEVICE)

        def run_cond(make_model, fwd):
            ms, first = [], None
            for s in range(CV_CONFIG["n_seeds"]):
                set_seed(CV_CONFIG["seed"] + 100 * s + fold_i)
                m, th, pr = train_one_model(
                    make_model(), fwd, y, train_mask, val_mask, test_mask,
                    pos_weight, CV_CONFIG, GNN_DEVICE,
                )
                ms.append(m)
                if first is None:
                    first = (th, pr)
            return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}, first

        conds = {
            "distilroberta_only": (lambda: DistilRobertaOnlyClassifier(text_dim=td),
                                   lambda m: m(x_text)),
            "text_tabular":       (lambda: TextTabularMLP(td, tt),
                                   lambda m: m(x_text, x_tab)),
            "graphsage_only":     (lambda: GraphSAGEOnlyBaseline(num_nodes=num_nodes, rand_dim=td,
                                                                 seed=CV_CONFIG["seed"]),
                                   lambda m: m(edge_index)),
            "late_concat":        (lambda: FraudDetectionGraphSAGE(td, tt, fusion_mode="concat"),
                                   lambda m: m(x_text, edge_index, x_tab)),
            "cross_attention":    (lambda: FraudDetectionGraphSAGE(td, tt, fusion_mode="cross_attention"),
                                   lambda m: m(x_text, edge_index, x_tab)),
        }

        for name, (mk, fwd) in conds.items():
            avg, (thresh, probs) = run_cond(mk, fwd)
            results[name].append(avg)
            print(f"[{name:18s}] F1={avg['f1']:.4f} F2={avg['f2']:.4f} PR-AUC={avg['pr_auc']:.4f}")
            if name == "cross_attention":
                results.setdefault("connectivity", []).append(
                    connectivity_breakdown(probs, thresh, y, test_mask, edge_index)
                )

        # save after every fold so a crash doesn't lose finished folds
        out_path = f"outputs/cv_results_{CV_CONFIG['split']}{CV_CONFIG['tag']}_seed{CV_CONFIG['seed']}.json"
        with open(out_path, "w") as f:
            json.dump({"config": CV_CONFIG, "timestamp": time.ctime(), "results": results}, f, indent=2)

    print("\n===== SUMMARY (mean ± std across folds) =====")
    summary_metrics = ["f1", "f2", "precision", "recall", "pr_auc", "mcc"]
    for cond in CONDITIONS:
        parts = []
        for m in summary_metrics:
            vals = [r[m] for r in results[cond]]
            parts.append(f"{m}={np.mean(vals):.3f}±{np.std(vals):.3f}")
        print(f"{cond:20s} " + "  ".join(parts))


if __name__ == "__main__":
    run()