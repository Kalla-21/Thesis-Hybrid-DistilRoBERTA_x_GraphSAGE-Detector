# train_cv.py
from logger import RunLogger  # keep first: configures library logging before transformers loads

import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, fbeta_score,
    matthews_corrcoef, precision_score, recall_score,
)

from losses import FocalLoss
from models.distilroberta_encoder import DistilRobertaModule
from models.graphsage_model import (
    DistilRobertaOnlyClassifier, FraudDetectionGraphSAGE,
    GraphSAGEOnlyBaseline, TextTabularMLP,
)
from utils import (
    DEVICE, GNN_DEVICE, HARDWARE, connectivity_breakdown, load_data,
    set_seed, stratified_kfold_masks,
)

CV_CONFIG = {
    "k_folds":         5,
    "seed":            42,
    "split":           "random",   # "random" or "group"
    "n_seeds":         3,          # model seeds averaged inside every fold
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
    "tag":             "",
}
CONDITIONS = ["distilroberta_only", "text_tabular", "graphsage_only", "late_concat", "cross_attention"]
SUMMARY_METRICS = ["f1", "f2", "precision", "recall", "pr_auc", "mcc"]


def tune_threshold(probs, labels, steps=80, beta=1.0, t_min=0.02):
    """Pick the decision threshold with the best (lightly smoothed) F-beta on validation data."""
    grid = np.linspace(t_min, 0.9, steps)
    scores = np.array([
        fbeta_score(labels, (probs >= t).astype(float), beta=beta, zero_division=0) for t in grid
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


def compute_pos_weight(y, train_mask, cap):
    num_licit = (y[train_mask] == 0).sum().item()
    num_scam = (y[train_mask] == 1).sum().item()
    return torch.tensor([min(num_licit / max(num_scam, 1), cap)], dtype=torch.float)


def make_conditions(x_text, x_tab, edge_index, seed):
    """The five models, each as (model factory, forward function)."""
    td, tt = x_text.shape[1], x_tab.shape[1]
    num_nodes = x_text.shape[0]
    return {
        "distilroberta_only": (lambda: DistilRobertaOnlyClassifier(text_dim=td),
                               lambda m: m(x_text)),
        "text_tabular":       (lambda: TextTabularMLP(td, tt),
                               lambda m: m(x_text, x_tab)),
        "graphsage_only":     (lambda: GraphSAGEOnlyBaseline(num_nodes=num_nodes, rand_dim=td, seed=seed),
                               lambda m: m(edge_index)),
        "late_concat":        (lambda: FraudDetectionGraphSAGE(td, tt, fusion_mode="concat"),
                               lambda m: m(x_text, edge_index, x_tab)),
        "cross_attention":    (lambda: FraudDetectionGraphSAGE(td, tt, fusion_mode="cross_attention"),
                               lambda m: m(x_text, edge_index, x_tab)),
    }


def train_one_model(model, forward_fn, y, train_mask, val_mask, test_mask,
                    pos_weight, cfg, device):
    """Full-batch training with early stopping on validation PR-AUC. The best checkpoint is
    loaded back into `model`; returns (test metrics, validation-tuned threshold, probabilities for all nodes)."""
    model = model.to(device)
    criterion = FocalLoss(gamma=cfg["gamma"], pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg["lr_factor"], patience=cfg["lr_patience"]
    )

    best_val, best_state, patience_count = 0.0, None, 0

    for _ in range(cfg["epochs"]):
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

    if best_state is None:  # validation score never improved: keep the last epoch
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


def get_embeddings(df, col, max_len, log):
    """DistilRoBERTa embeddings of df[col], cached on disk. The cache name contains a hash of the
    texts, so changing the preprocessing automatically invalidates the cache."""
    texts = df[col].tolist()
    key = hashlib.sha1("\x1f".join(texts).encode("utf-8")).hexdigest()[:10]
    path = os.path.join("outputs", f"emb_{col}_{max_len}_{key}.npy")
    if os.path.exists(path):
        log.event("embeddings", "loaded from cache", path=path)
        return np.load(path)
    t0 = time.time()
    emb = DistilRobertaModule(DEVICE).embed_texts(texts, max_len=max_len)
    np.save(path, emb)
    log.event("embeddings", "computed and cached", path=path, shape=list(emb.shape),
              seconds=round(time.time() - t0, 1))
    return emb


def load_inputs(log):
    """Shared by train_cv.py and build_model.py: dataframe + all model inputs as tensors."""
    os.makedirs("outputs", exist_ok=True)
    log.event("system", "hardware", **HARDWARE)
    df, edges_arr, tabular_features = load_data(log)
    embeddings = get_embeddings(df, "full_text", 512, log)
    x_text = torch.tensor(embeddings, dtype=torch.float).to(GNN_DEVICE)
    x_tab = torch.tensor(tabular_features, dtype=torch.float).to(GNN_DEVICE)
    y = torch.tensor(df["label"].values, dtype=torch.float).to(GNN_DEVICE)
    edge_index = torch.tensor(edges_arr, dtype=torch.long).to(GNN_DEVICE)
    return df, x_text, x_tab, y, edge_index


def _new_oof(labels, k):
    n = len(labels)
    oof = {"y": labels.astype(np.int8), "fold": np.full(n, -1, dtype=np.int8)}
    for cond in CONDITIONS:
        oof[f"probs_{cond}"] = np.zeros(n, dtype=np.float32)
        oof[f"thresh_{cond}"] = np.zeros(k, dtype=np.float32)
    return oof


def run(split="random", seed=42):
    cfg = {**CV_CONFIG, "split": split, "seed": seed}
    with RunLogger(f"train_cv_{split}_seed{seed}") as log:
        set_seed(seed)
        df, x_text, x_tab, y, edge_index = load_inputs(log)
        labels = df["label"].values
        groups = df["group_id"].values if split == "group" else None

        results = {cond: [] for cond in CONDITIONS}
        oof = _new_oof(labels, cfg["k_folds"])
        results_path = os.path.join("outputs", f"cv_results_{split}{cfg['tag']}_seed{seed}.json")
        oof_path = os.path.join("outputs", f"oof_{split}{cfg['tag']}_seed{seed}.npz")
        log.event("cv", "starting", split=split, seed=seed, config=cfg)

        for fold_i, train_mask, val_mask, test_mask in stratified_kfold_masks(
            labels, k=cfg["k_folds"], seed=seed, groups=groups
        ):
            train_mask, val_mask, test_mask = (m.to(GNN_DEVICE) for m in (train_mask, val_mask, test_mask))
            pos_weight = compute_pos_weight(y, train_mask, cfg["pos_weight_cap"]).to(GNN_DEVICE)
            test_np = test_mask.cpu().numpy()
            conds = make_conditions(x_text, x_tab, edge_index, seed)

            for name, (make_model, forward_fn) in conds.items():
                per_seed, first = [], None
                for s in range(cfg["n_seeds"]):
                    set_seed(seed + 100 * s + fold_i)
                    metrics, thresh, probs = train_one_model(
                        make_model(), forward_fn, y, train_mask, val_mask, test_mask,
                        pos_weight, cfg, GNN_DEVICE,
                    )
                    per_seed.append(metrics)
                    if first is None:
                        first = (thresh, probs)
                avg = {k: float(np.mean([m[k] for m in per_seed])) for k in per_seed[0]}
                thresh, probs = first
                results[name].append(avg)

                # out-of-fold predictions (first model seed): every posting is a test posting exactly once
                oof["fold"][test_np] = fold_i
                oof[f"probs_{name}"][test_np] = probs[test_np]
                oof[f"thresh_{name}"][fold_i] = thresh

                if name == "cross_attention":
                    results.setdefault("connectivity", []).append(
                        connectivity_breakdown(probs, thresh, y, test_mask, edge_index)
                    )
                log.event("fold", f"{name} done", fold=fold_i + 1, condition=name,
                          f1=avg["f1"], f2=avg["f2"], mcc=avg["mcc"], pr_auc=avg["pr_auc"])

            # saved after every fold, so a crash keeps the finished folds
            with open(results_path, "w") as f:
                json.dump({"config": cfg, "timestamp": time.ctime(), "results": results}, f, indent=2)
            np.savez_compressed(oof_path, **oof)

        summary = {
            cond: {m: {"mean": float(np.mean([r[m] for r in results[cond]])),
                       "std": float(np.std([r[m] for r in results[cond]]))}
                   for m in SUMMARY_METRICS}
            for cond in CONDITIONS
        }
        log.finish("ok", results_file=results_path, oof_file=oof_path, cv_summary=summary)


if __name__ == "__main__":
    _split = sys.argv[1] if len(sys.argv) > 1 else CV_CONFIG["split"]
    _seed = int(sys.argv[2]) if len(sys.argv) > 2 else CV_CONFIG["seed"]
    assert _split in ("random", "group"), "split must be 'random' or 'group'"
    run(_split, _seed)