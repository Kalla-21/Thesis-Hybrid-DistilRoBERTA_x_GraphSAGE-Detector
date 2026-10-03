"""One-off: export the training graph + decision threshold into model/ for the demo.

Run with the TRAINING repo's python (it needs pandas / sklearn / torch_geometric):

    <train_repo>\\.venv\\Scripts\\python.exe build_graph_assets.py --train-repo <train_repo>

Writes:
    model/graph_assets.npz        embeddings, tabular feats, edges, company/title hashes
    model/inference_config.json   tuned threshold + the settings the model was trained with
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def h64(s: str) -> np.uint64:
    """Stable 64-bit hash of an exact string; 0 is reserved for the empty string."""
    if s == "":
        return np.uint64(0)
    return np.frombuffer(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), dtype=np.uint64)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-repo", required=True)
    args = ap.parse_args()

    train = os.path.abspath(args.train_repo)
    os.chdir(train)  # training code uses relative paths (data/, outputs/)
    sys.path.insert(0, train)
    from utils import load_and_split_data, stratified_kfold_masks
    from train_cv import tune_threshold, get_metrics
    from models.graphsage_model import FraudDetectionGraphSAGE

    df, edges, _, _, _, tab = load_and_split_data()
    y = df["label"].values
    emb = np.load("outputs/embeddings.npy")
    assert emb.shape == (len(df), 768)

    # --- which CV fold produced the shipped weights? (the one with the best cross_attention F1)
    cv = json.load(open("outputs/cv_results.json"))["cross_attention"]
    fold_idx = int(np.argmax([r["f1"] for r in cv]))
    recorded_f1 = cv[fold_idx]["f1"]

    weights = os.path.join(HERE, "model", "best_cross_attention_model.pt")
    model = FraudDetectionGraphSAGE(768, 3, fusion_mode="cross_attention")
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
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
        thresh, val_f1 = tune_threshold(probs[va], y[va], 50)
        m = get_metrics(y[te], (probs[te] >= thresh).astype(float))
    print(f"fold {fold_idx}: threshold={thresh:.5f}  held-out F1={m['f1']:.4f} (cv_results.json: {recorded_f1:.4f})")
    assert abs(m["f1"] - recorded_f1) < 1e-3, "weights/embeddings don't reproduce the recorded CV result"

    company_hash = np.array([h64(s) for s in df["company_profile"].values], dtype=np.uint64)
    title_hash = np.array([h64(s) for s in df["title"].values], dtype=np.uint64)

    np.savez_compressed(
        os.path.join(HERE, "model", "graph_assets.npz"),
        embeddings=emb.astype(np.float16),  # halves the file; effect on outputs is checked below
        tabular=tab.astype(np.float32),
        edge_index=edges.astype(np.int32),
        company_hash=company_hash,
        title_hash=title_hash,
    )
    with open(os.path.join(HERE, "model", "inference_config.json"), "w") as f:
        json.dump({
            "threshold": thresh,
            "max_length": 256,
            "source_fold": fold_idx,
            "heldout_f1_of_source_fold": m["f1"],
            "cv_mean_f1": float(np.mean([r["f1"] for r in cv])),
            "max_title_group": 15,
        }, f, indent=2)

    # float16 sanity check: do verdicts change vs the float32 embeddings?
    with torch.no_grad():
        p16 = torch.sigmoid(model(
            torch.tensor(emb.astype(np.float16).astype(np.float32)),
            torch.tensor(edges, dtype=torch.long),
            torch.tensor(tab, dtype=torch.float),
        )).squeeze().numpy()
    print(f"float16 embeddings: max |dprob|={np.abs(p16 - probs).max():.5f}, "
          f"verdict flips={int(((p16 >= thresh) != (probs >= thresh)).sum())} of {len(probs)}")


if __name__ == "__main__":
    main()
