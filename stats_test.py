# stats_test.py
import glob, json
import numpy as np
from scipy import stats

PRIMARY = "cross_attention"
HYPOTHESES = [
    ("H1: hybrid vs DistilRoBERTa-only", "distilroberta_only"),
    ("H2: hybrid vs GraphSAGE-only",     "graphsage_only"),
    ("H3: cross-attention vs late fusion", "late_concat"),
]
METRICS = ["f1", "mcc"]
ALPHA = 0.05


def load(split):
    files = sorted(glob.glob(f"outputs/cv_results_{split}_seed*.json"))
    res = {}
    for fp in files:
        with open(fp) as f:
            r = json.load(f)["results"]
        for cond, folds in r.items():
            if cond != "connectivity":
                res.setdefault(cond, []).extend(folds)
    return res, len(files)


def test(a, b):
    a, b = np.array(a), np.array(b)
    d = a - b
    if np.allclose(d, d[0]):
        return d, "n/a", np.nan, np.nan
    sw_p = stats.shapiro(d)[1] if len(d) >= 3 else 0.0
    if sw_p > ALPHA:
        stat, p = stats.ttest_rel(a, b); name = "paired t-test"
    else:
        try:
            stat, p = stats.wilcoxon(a, b)
        except ValueError:
            stat, p = np.nan, np.nan
        name = "Wilcoxon signed-rank"
    return d, name, float(stat), float(p)


def run():
    for split in ("random", "group"):
        res, n_files = load(split)
        if not res:
            continue
        rows = []
        for label, base in HYPOTHESES:
            for m in METRICS:
                a = [r[m] for r in res[PRIMARY]]
                b = [r[m] for r in res[base]]
                rows.append((label, m, *test(a, b)))

        print(f"\n##### {split.upper()} SPLIT ({n_files} CV runs, n={len(rows and rows[0][2])} paired values) #####")
        for label, m, d, name, stat, p in rows:
            half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            # Verdict now strictly uses the unadjusted p-value
            verdict = "REJECT H0" if (p == p and p < ALPHA) else "fail to reject H0"
            print(f"{label:38s} {m.upper():4s} diff={d.mean():+.4f} "
                  f"[95% CI {d.mean()-half:+.4f}, {d.mean()+half:+.4f}] | "
                  f"{name}: p={p:.4f} -> {verdict}")

    print("\nNote: folds share training data, so p-values are approximate guides.")


if __name__ == "__main__":
    run()