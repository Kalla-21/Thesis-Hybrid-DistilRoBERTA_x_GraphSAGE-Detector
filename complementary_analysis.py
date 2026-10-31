# complementary_analysis.py
from logger import RunLogger

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NEEDED = ["distilroberta_only", "graphsage_only", "late_concat", "cross_attention"]
COST_KEYS = {"distilroberta_only": ("text_hits", "fp_text"), "graphsage_only": ("graph_hits", "fp_graph"),
             "late_concat": ("late_hits", "fp_late"), "cross_attention": ("hybrid_hits", "fp_hybrid")}


def run(split, seed):
    with RunLogger(f"complementary_{split}_seed{seed}") as log:
        oof_path = os.path.join("outputs", f"oof_{split}_seed{seed}.npz")
        if not os.path.exists(oof_path):
            raise FileNotFoundError(f"{oof_path} not found. Run train_cv.py {split} {seed} first.")
        oof = np.load(oof_path)
        y, fold = oof["y"], oof["fold"]
        k = int(oof["thresh_late_concat"].shape[0])

        keys = ["n_fraud", "neither", "text_only", "graph_only", "both", "hybrid_of_neither", "late_of_neither",
                "text_hits", "graph_hits", "late_hits", "hybrid_hits",
                "n_legit", "fp_text", "fp_graph", "fp_late", "fp_hybrid"]
        tot = dict.fromkeys(keys, 0)
        per_fold = []

        for f in range(k):
            test = fold == f
            fraud = y[test] == 1
            legit = ~fraud
            pred = {name: oof[f"probs_{name}"][test] >= oof[f"thresh_{name}"][f] for name in NEEDED}

            dr, gs, lc, ca = (pred[name][fraud] for name in NEEDED)
            neither = ~dr & ~gs
            row = dict(
                fold=f + 1, n_fraud=int(fraud.sum()), neither=int(neither.sum()),
                text_only=int((dr & ~gs).sum()), graph_only=int((~dr & gs).sum()), both=int((dr & gs).sum()),
                hybrid_of_neither=int((ca & neither).sum()), late_of_neither=int((lc & neither).sum()),
                text_hits=int(dr.sum()), graph_hits=int(gs.sum()), late_hits=int(lc.sum()), hybrid_hits=int(ca.sum()),
                n_legit=int(legit.sum()),
                fp_text=int(pred["distilroberta_only"][legit].sum()), fp_graph=int(pred["graphsage_only"][legit].sum()),
                fp_late=int(pred["late_concat"][legit].sum()), fp_hybrid=int(pred["cross_attention"][legit].sum()),
            )
            per_fold.append(row)
            for key in keys:
                tot[key] += row[key]

        n = tot["n_fraud"]
        cost = {}
        for name, (hits_key, fp_key) in COST_KEYS.items():
            cost[name] = {"false_alarms": tot[fp_key],
                          "precision": tot[hits_key] / max(tot[hits_key] + tot[fp_key], 1),
                          "false_alarm_rate": tot[fp_key] / max(tot["n_legit"], 1)}

        report = {
            "split": split, "pooled": tot, "per_fold": per_fold, "cost": cost,
            "recall_distilroberta_only": tot["text_hits"] / n,
            "recall_graphsage_only": tot["graph_hits"] / n,
            "recall_late_concat": tot["late_hits"] / n,
            "recall_cross_attention": tot["hybrid_hits"] / n,
            "recovered_of_missed_by_both": {
                "missed_by_both": tot["neither"],
                "cross_attention": tot["hybrid_of_neither"],
                "late_concat": tot["late_of_neither"],
            },
        }
        report_path = os.path.join("outputs", f"complementarity_report_{split}.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        labels = ["Neither\nstandalone", "Text\nonly", "Graph\nonly", "Both\nstandalone"]
        counts = [tot["neither"], tot["text_only"], tot["graph_only"], tot["both"]]
        plt.figure(figsize=(6, 5))
        bars = plt.bar(labels, counts, color=["#c0392b", "#2980b9", "#27ae60", "#8e44ad"])
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, str(int(count)), ha="center")
        plt.ylabel("Fraudulent test listings (pooled over folds)")
        plt.title(f"Which standalone model catches which fraud ({split} split)")
        plt.tight_layout()
        plt.savefig(os.path.join("outputs", f"complementarity_breakdown_{split}.png"))
        plt.close()

        log.finish("ok", report_file=report_path, n_fraud=n, **{
            "recall": {name: report[f"recall_{name}"] for name in NEEDED}})


if __name__ == "__main__":
    _split = sys.argv[1] if len(sys.argv) > 1 else "random"
    _seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    assert _split in ("random", "group"), "split must be 'random' or 'group'"
    run(_split, _seed)