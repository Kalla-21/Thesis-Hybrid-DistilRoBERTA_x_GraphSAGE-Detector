"""Bridge script: exports training graph + best model weights into model/ for UI inference.

Run after training finishes:
    python build_graph_assets.py
"""
import glob
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from utils import load_and_split_data, stratified_kfold_masks
from train_cv import tune_threshold, get_metrics
from models.graphsage_model import FraudDetectionGraphSAGE


def h64(s: str) -> np.uint64:
    if s == "":
        return np.uint64(0)
    return np.frombuffer(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), dtype=np.uint64)[0]


def main():
    os.makedirs(os.path.join(HERE, "model"), exist_ok=True)

    # 1. Load dataset & tabular features from training logic
    df, edges, _, _, _, tab = load_and_split_data()
    y = df["label"].values
    tt = tab.shape[1]  # Dynamically obtain tabular feature dimension (11)

    emb_path = "outputs/emb_full_512.npy"
    if not os.path.exists(emb_path):
        emb_path = "outputs/embeddings.npy"
    emb = np.load(emb_path)

    # 2. Locate CV results JSON log
    cv_files = sorted(glob.glob("outputs/cv_results_*.json"))
    if not cv_files:
        sys.exit("Error: No cv_results_*.json file found in outputs/. Run train_cv.py first.")
    
    cv_file = cv_files[0]
    print(f"Reading CV results from {cv_file}...")
    
    with open(cv_file) as f:
        data = json.load(f)
        cv = data["results"]["cross_attention"] if "results" in data else data["cross_attention"]

    # 3. Copy best fold checkpoint
    fold_idx = int(np.argmax([r["f1"] for r in cv]))
    recorded_f1 = cv[fold_idx]["f1"]
    best_fold_number = fold_idx + 1
    
    source_weights = f"outputs/cross_attention_fold_{best_fold_number}.pt"
    dest_weights = os.path.join(HERE, "model", "best_cross_attention_model.pt")
    shutil.copy(source_weights, dest_weights)
    print(f"Selected fold {best_fold_number} (F1={recorded_f1:.4f}) -> Copied to {dest_weights}")

    # 4. Instantiate model with matching tabular dimension (tt)
    model = FraudDetectionGraphSAGE(768, tt, fusion_mode="cross_attention")
    model.load_state_dict(torch.load(dest_weights, map_location="cpu", weights_only=True))
    model.eval()

    with torch.no_grad():
        probs = torch.sigmoid(model(
            torch.tensor(emb, dtype=torch.float),
            torch.tensor(edges, dtype=torch.long),
            torch.tensor(tab, dtype=torch.float),
        )).squeeze().numpy()

    for fold, _tr, va, te in stratified_kfold_masks(y, 5, 42):
        if fold != fold_idx:
            continue
        va, te = va.numpy(), te.numpy()
        thresh, _ = tune_threshold(probs[va], y[va], 80, beta=1.0)
        m = get_metrics(y[te], (probs[te] >= thresh).astype(float))

    print(f"Fold {best_fold_number} tuned threshold={thresh:.5f} | Held-out F1={m['f1']:.4f}")

    # 5. Export graph assets & config with correct tabular_dim
    company_hash = np.array([h64(s) for s in df["company_profile"].values], dtype=np.uint64)
    title_hash = np.array([h64(s) for s in df["title"].values], dtype=np.uint64)

    np.savez_compressed(
        os.path.join(HERE, "model", "graph_assets.npz"),
        embeddings=emb.astype(np.float16),
        tabular=tab.astype(np.float32),
        edge_index=edges.astype(np.int32),
        company_hash=company_hash,
        title_hash=title_hash,
    )

    with open(os.path.join(HERE, "model", "inference_config.json"), "w") as f:
        json.dump({
            "threshold": float(thresh),
            "max_length": 256,
            "tabular_dim": int(tt),
            "source_fold": best_fold_number,
            "heldout_f1_of_source_fold": float(m["f1"]),
            "cv_mean_f1": float(np.mean([r["f1"] for r in cv])),
            "max_title_group": 15,
        }, f, indent=2)

    print("Graph assets and inference config exported successfully!")


if __name__ == "__main__":
    main()