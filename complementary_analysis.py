# complementary_analysis.py
from logger import RunLogger

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# This script performs a complementary analysis of the model's performance on different splits of the dataset (random and group).
NEEDED = ["distilroberta_only", "graphsage_only", "late_concat", "cross_attention"]
COST_KEYS = {"distilroberta_only": ("text_hits", "fp_text"), "graphsage_only": ("graph_hits", "fp_graph"),
             "late_concat": ("late_hits", "fp_late"), "cross_attention": ("hybrid_hits", "fp_hybrid")}


def run(split, seed):
    # This function performs the complementary analysis for a given split (random or group) and seed.
    # It loads the out-of-fold predictions, computes various metrics, and generates a report and visualizations.
    with RunLogger(f"complementary_{split}_seed{seed}") as log:
        # Load the out-of-fold predictions from the specified split and seed, raising an error if the file is not found.
        oof_path = os.path.join("outputs", f"oof_{split}_seed{seed}.npz")
        if not os.path.exists(oof_path):
            raise FileNotFoundError(f"{oof_path} not found. Run train_cv.py {split} {seed} first.")
        oof = np.load(oof_path)
        y, fold = oof["y"], oof["fold"]
        k = int(oof["thresh_late_concat"].shape[0])
        
        # Compute the total counts and per-fold counts for various metrics, including the number of fraudulent and legitimate postings,
        # the number of postings caught by each model, and the number of false positives for each model. 
        # The results are stored in the report dictionary, which is saved as a JSON file.
        keys = ["n_fraud", "neither", "text_only", "graph_only", "both", "hybrid_of_neither", "late_of_neither",
                "text_hits", "graph_hits", "late_hits", "hybrid_hits",
                "n_legit", "fp_text", "fp_graph", "fp_late", "fp_hybrid"]
        tot = dict.fromkeys(keys, 0)
        per_fold = []

        # Loop through each fold of the cross-validation, computing the metrics for that fold and updating the total counts.
        for f in range(k):
            test = fold == f
            fraud = y[test] == 1
            legit = ~fraud
            pred = {name: oof[f"probs_{name}"][test] >= oof[f"thresh_{name}"][f] for name in NEEDED}

            # Compute the counts for each model and the number of postings that were caught by neither model, only one model, or both models.
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
            
            # Update the total counts for each metric by adding the counts from the current fold to the running totals.
            per_fold.append(row)
            for key in keys:
                tot[key] += row[key]

        # Save the report as a JSON file in the "outputs" directory, 
        # along with CSV files for detailed metrics and connectivity information if available.
        n = tot["n_fraud"]
        cost = {}
        for name, (hits_key, fp_key) in COST_KEYS.items():
            cost[name] = {"false_alarms": tot[fp_key],
                          "precision": tot[hits_key] / max(tot[hits_key] + tot[fp_key], 1),
                          "false_alarm_rate": tot[fp_key] / max(tot["n_legit"], 1)}

        report = { # The report dictionary contains the results of the complementary analysis,
                  #including the split type, total counts, per-fold counts, and cost metrics for each model.
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
        with open(report_path, "w") as f: # Save the report as a JSON file in the "outputs" directory, with a name based on the split type.
            json.dump(report, f, indent=2)

        # Generate a bar chart showing the counts of fraudulent test listings caught by each model,
        # including the number of listings caught by neither model, only one model, or both models. 
        # The chart is saved as a PNG file in the "outputs" directory, with a name based on the split type.
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

        # Log the completion of the complementary analysis, including the path to the report file, 
        # the total number of fraudulent listings, and the recall metrics for each model.
        log.finish("ok", report_file=report_path, n_fraud=n, **{
            "recall": {name: report[f"recall_{name}"] for name in NEEDED}})


if __name__ == "__main__":
    _split = sys.argv[1] if len(sys.argv) > 1 else "random"
    _seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    assert _split in ("random", "group"), "split must be 'random' or 'group'"
    run(_split, _seed)