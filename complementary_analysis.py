# complementary_analysis.py
import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import GNN_DEVICE, set_seed, stratified_kfold_masks
from train_cv import (train_one_model, CV_CONFIG, make_conditions,
                      compute_pos_weight, load_inputs)

NEEDED = ["distilroberta_only", "graphsage_only", "late_concat", "cross_attention"]


def run(split):
    os.makedirs("outputs", exist_ok=True)
    seed = CV_CONFIG["seed"]
    set_seed(seed)

    df, x_text, x_tab, y, edge_index = load_inputs()
    y_np = y.cpu().numpy()
    labels_arr = df["label"].values
    groups = df["group_id"].values if split == "group" else None

    tot = dict(n_fraud=0, neither=0, text_only=0, graph_only=0, both=0,
               hybrid_of_neither=0, late_of_neither=0,
               text_hits=0, graph_hits=0, late_hits=0, hybrid_hits=0,
               n_legit=0, fp_text=0, fp_graph=0, fp_late=0, fp_hybrid=0)
    per_fold = []

    for fold_i, tr, va, te in stratified_kfold_masks(labels_arr, k=CV_CONFIG["k_folds"], seed=seed, groups=groups):
        tr, va, te = tr.to(GNN_DEVICE), va.to(GNN_DEVICE), te.to(GNN_DEVICE)
        pos_weight = compute_pos_weight(y, tr, CV_CONFIG["pos_weight_cap"]).to(GNN_DEVICE)
        conds = make_conditions(x_text, x_tab, edge_index, seed)

        preds = {}
        for name in NEEDED:
            mk, fwd = conds[name]
            set_seed(seed + fold_i)   # same seed as the first model seed in train_cv.py
            _, th, pr = train_one_model(mk(), fwd, y, tr, va, te, pos_weight, CV_CONFIG, GNN_DEVICE)
            preds[name] = (pr >= th).astype(int)
            print(f"[fold {fold_i + 1}] trained {name}")

        te_np = te.cpu().numpy()
        fraud_idx = np.where(te_np & (y_np == 1))[0]
        legit_idx = np.where(te_np & (y_np == 0))[0]

        dr = preds["distilroberta_only"][fraud_idx] == 1
        gs = preds["graphsage_only"][fraud_idx] == 1
        lc = preds["late_concat"][fraud_idx] == 1
        ca = preds["cross_attention"][fraud_idx] == 1
        neither = ~dr & ~gs

        # false alarms: legitimate test listings each model flags as fraud
        fp = {k: int(preds[k][legit_idx].sum()) for k in NEEDED}

        row = dict(fold=fold_i + 1, n_fraud=len(fraud_idx), neither=int(neither.sum()),
                   text_only=int((dr & ~gs).sum()), graph_only=int((~dr & gs).sum()), both=int((dr & gs).sum()),
                   hybrid_of_neither=int((ca & neither).sum()), late_of_neither=int((lc & neither).sum()),
                   text_hits=int(dr.sum()), graph_hits=int(gs.sum()), late_hits=int(lc.sum()), hybrid_hits=int(ca.sum()),
                   n_legit=len(legit_idx), fp_text=fp["distilroberta_only"], fp_graph=fp["graphsage_only"],
                   fp_late=fp["late_concat"], fp_hybrid=fp["cross_attention"])
        per_fold.append(row)
        for k in tot:
            tot[k] += row[k]

    n = tot["n_fraud"]

    # cost of the detections: precision and false-alarm rate, pooled over folds
    cost = {}
    for name, hits_key, fp_key in [("distilroberta_only", "text_hits", "fp_text"),
                                   ("graphsage_only", "graph_hits", "fp_graph"),
                                   ("late_concat", "late_hits", "fp_late"),
                                   ("cross_attention", "hybrid_hits", "fp_hybrid")]:
        cost[name] = {"false_alarms": tot[fp_key],
                      "precision": tot[hits_key] / max(tot[hits_key] + tot[fp_key], 1),
                      "false_alarm_rate": tot[fp_key] / max(tot["n_legit"], 1)}

    report = {"split": split, "pooled": tot, "per_fold": per_fold, "cost": cost,
              "recall_distilroberta_only": tot["text_hits"] / n,
              "recall_graphsage_only": tot["graph_hits"] / n,
              "recall_late_concat": tot["late_hits"] / n,
              "recall_cross_attention": tot["hybrid_hits"] / n}
    with open(f"outputs/complementarity_report_{split}.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n===== COMPLEMENTARITY ({split} split, pooled over folds; {n} fraudulent test listings) =====")
    print(f"Caught by neither standalone : {tot['neither']}")
    print(f"Caught by text only          : {tot['text_only']}")
    print(f"Caught by graph only         : {tot['graph_only']}")
    print(f"Caught by both               : {tot['both']}")
    if tot["neither"]:
        print(f"Of the {tot['neither']} missed by both: cross-attention catches {tot['hybrid_of_neither']} "
              f"({100 * tot['hybrid_of_neither'] / tot['neither']:.1f}%), late concat catches {tot['late_of_neither']} "
              f"({100 * tot['late_of_neither'] / tot['neither']:.1f}%)")
    print(f"Recall: text {report['recall_distilroberta_only']:.3f} | graph {report['recall_graphsage_only']:.3f} | "
          f"late {report['recall_late_concat']:.3f} | cross-attn {report['recall_cross_attention']:.3f}")
    print("Cost (pooled): precision and false alarms among legitimate listings")
    for name, v in cost.items():
        print(f"  {name:18s} precision {v['precision']:.3f} | false alarms {v['false_alarms']} "
              f"({100 * v['false_alarm_rate']:.2f}% of {tot['n_legit']} legitimate)")

    labels = ["Neither\nstandalone", "Text\nonly", "Graph\nonly", "Both\nstandalone"]
    counts = [tot["neither"], tot["text_only"], tot["graph_only"], tot["both"]]
    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, counts, color=["#c0392b", "#2980b9", "#27ae60", "#8e44ad"])
    for b, c in zip(bars, counts):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, str(int(c)), ha="center")
    plt.ylabel("Fraudulent test listings (pooled over folds)")
    plt.title(f"Which standalone model catches which fraud ({split} split)")
    plt.tight_layout()
    plt.savefig(f"outputs/complementarity_breakdown_{split}.png")
    plt.close()


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else CV_CONFIG["split"]
    assert split in ("random", "group")
    run(split)