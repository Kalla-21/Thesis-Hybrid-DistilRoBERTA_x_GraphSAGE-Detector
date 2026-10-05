# stats_test.py
import os
import json
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

RESULT_FILES = {
    "random split": "outputs/cv_results_random.json",
    "group split":  "outputs/cv_results_group.json",
}
PRIMARY = "cross_attention"
BASELINES = ["distilroberta_only", "text_tabular", "graphsage_only", "late_concat"]
METRICS = ["f1", "f2", "recall", "pr_auc", "mcc"]
ALPHA = 0.05


def paired_test(primary_scores, baseline_scores):
    a, b = np.array(primary_scores), np.array(baseline_scores)
    diffs = a - b
    if np.allclose(diffs, diffs[0]):
        return diffs, float("nan"), float("nan")
    stat, p = stats.ttest_rel(a, b)
    return diffs, float(stat), float(p)


def run_one(label, path):
    with open(path) as f:
        results = json.load(f)["results"]

    rows = []
    for baseline in BASELINES:
        for metric in METRICS:
            a = [fold[metric] for fold in results[PRIMARY]]
            b = [fold[metric] for fold in results[baseline]]
            diffs, stat, p = paired_test(a, b)
            rows.append((baseline, metric, diffs, stat, p))

    # Holm correction across all comparisons in this file
    pvals = np.array([r[4] for r in rows])
    valid = ~np.isnan(pvals)
    adj = np.full(len(rows), np.nan)
    if valid.any():
        adj[valid] = multipletests(pvals[valid], method="holm")[1]

    print(f"\n########## {label.upper()} ({len(rows)} comparisons, Holm-corrected) ##########")
    current = None
    for (baseline, metric, diffs, stat, p), pa in zip(rows, adj):
        if baseline != current:
            print(f"\n=== {PRIMARY} vs {baseline} ===")
            current = baseline
        verdict = "SIGNIFICANT" if (pa == pa and pa < ALPHA) else "not significant"
        print(f"    {metric.upper():7s} | mean diff = {diffs.mean():+.4f} "
              f"(sd={diffs.std(ddof=1):.4f}, n={len(diffs)}) | "
              f"t={stat:.3f}, p={p:.4f}, holm p={pa:.4f} -> {verdict}")


def run():
    found = False
    for label, path in RESULT_FILES.items():
        if os.path.exists(path):
            found = True
            run_one(label, path)
    if not found:
        print("No result files found in outputs/. Run train_cv.py first.")
    print("\nNote: with 5 folds, power is low and fold scores share training data, "
          "so treat p-values as rough guides, not proof.")


if __name__ == "__main__":
    run()