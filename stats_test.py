#stats_test.py
import json
import numpy as np
from scipy import stats

RESULTS_PATH = "outputs/cv_results.json"
PRIMARY = "cross_attention"
BASELINES = ["distilroberta_only", "graphsage_only", "late_concat"]
METRICS = ["f1", "mcc"]
ALPHA = 0.05


def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def compare(primary_scores, baseline_scores, metric_name):
    primary_scores = np.array(primary_scores)
    baseline_scores = np.array(baseline_scores)
    diffs = primary_scores - baseline_scores
    n = len(diffs)

    shapiro_p = None
    if n >= 3 and not np.allclose(diffs, diffs[0]):
        _, shapiro_p = stats.shapiro(diffs)

    if shapiro_p is not None and shapiro_p > ALPHA:
        stat, p = stats.ttest_rel(primary_scores, baseline_scores)
        test_used = "paired t-test"
    else:
        try:
            stat, p = stats.wilcoxon(primary_scores, baseline_scores)
        except ValueError:
            # Raised if all differences are 0, or n is too small for Wilcoxon.
            stat, p = float("nan"), float("nan")
        test_used = "Wilcoxon signed-rank"

    verdict = "SIGNIFICANT" if (p == p and p < ALPHA) else "not significant"
    print(f"    {metric_name.upper():5s} | mean diff = {diffs.mean():+.4f} "
          f"(sd={diffs.std():.4f}, n={n}) | {test_used}: stat={stat:.4f}, "
          f"p={p:.4f} -> {verdict}")


def run():
    results = load_results()
    for baseline in BASELINES:
        print(f"\n=== {PRIMARY} vs {baseline} ===")
        for metric in METRICS:
            primary_scores = [fold[metric] for fold in results[PRIMARY]]
            baseline_scores = [fold[metric] for fold in results[baseline]]
            compare(primary_scores, baseline_scores, metric)


if __name__ == "__main__":
    run()