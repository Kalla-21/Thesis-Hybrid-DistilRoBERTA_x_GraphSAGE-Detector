# stats_test.py
from logger import RunLogger

import csv
import glob
import json
import os

import numpy as np
from scipy import stats

PRIMARY = "cross_attention"
HYPOTHESES = [ # list of hypotheses to be tested, each represented as a tuple containing a description and the baseline condition for comparison.
    ("H1: hybrid vs DistilRoBERTa-only",   "distilroberta_only"),
    ("H2: hybrid vs GraphSAGE-only",       "graphsage_only"),
    ("H3: cross-attention vs late fusion", "late_concat"),
]
METRICS = ["f1", "mcc"] # list of metrics to be evaluated for each hypothesis, including F1 score and Matthews correlation coefficient (MCC).
ALPHA = 0.05 # significance level for statistical tests
TEST_TRAIN_RATIO = 0.29   # n_test / n_train per fold (0.20 / 0.68), for the Nadeau-Bengio corrected t-test


def load(split): # Loads the cross-validation results for a given split (random or group) 
                 # from JSON files in the "outputs" directory,
    res, conn, rows = {}, [], []
    files = sorted(glob.glob(os.path.join("outputs", f"cv_results_{split}_seed*.json")))
    for fp in files:
        with open(fp) as f:
            run = json.load(f)
        seed = run["config"]["seed"]
        for cond, folds in run["results"].items():
            if cond == "connectivity":
                for i, c in enumerate(folds):
                    for group, v in c.items():
                        conn.append(dict(split=split, seed=seed, fold=i + 1, group=group, **v))
                continue
            res.setdefault(cond, []).extend(folds)
            for i, m in enumerate(folds):
                for k, v in m.items():
                    rows.append(dict(split=split, seed=seed, fold=i + 1, condition=cond, metric=k, value=v))
    return res, len(files), rows, conn


def compare(a, b): # Compares two sets of paired values (a and b) using statistical tests, 
                   # including the Shapiro-Wilk test for normality,
                   # the paired t-test, and the Wilcoxon signed-rank test.
                   
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    sw_p = stats.shapiro(d)[1] if n >= 3 and not np.allclose(d, d[0]) else 0.0
    _, p_t = stats.ttest_rel(a, b)
    t_c = d.mean() / np.sqrt((1 / n + TEST_TRAIN_RATIO) * d.var(ddof=1))
    p_c = 2 * stats.t.sf(abs(t_c), n - 1)
    try:
        p_w = stats.wilcoxon(a, b)[1]
    except ValueError:
        p_w = float("nan")
    normal = sw_p > ALPHA
    return dict(d=d, n=n, shapiro_p=sw_p, p_t=p_t, p_corrected=p_c, p_wilcoxon=p_w,
                test="paired t-test" if normal else "Wilcoxon signed-rank",
                p=p_t if normal else p_w, wins=int((d > 0).sum()))


def run(): # Main function to run the statistical tests on the cross-validation results for both random and group splits.
    os.makedirs("outputs", exist_ok=True)
    with RunLogger("stats_test") as log:
        all_rows, all_conn, report = [], [], {"alpha": ALPHA, "splits": {}}

        # Loop through the two types of splits: "random" and "group", 
        # loading the results for each split and performing statistical tests on the paired values.
        for split in ("random", "group"):
            res, n_files, rows, conn = load(split)
            all_rows += rows
            all_conn += conn
            if not res:
                log.event("stats", "no result files found", split=split)
                continue
        
        # Compute the number of paired values for the primary condition (cross_attention) and perform statistical tests for
        # each hypothesis and metric,storing the results in the report dictionary along with notes about the tests and any limitations.
            n_pairs = len(res[PRIMARY])
            tests = []
            for label, base in HYPOTHESES:
                for metric in METRICS:
                    r = compare([x[metric] for x in res[PRIMARY]], [x[metric] for x in res[base]])
                    d = r["d"]
                    half = stats.t.ppf(0.975, r["n"] - 1) * d.std(ddof=1) / np.sqrt(r["n"])
                    significant = r["p"] < ALPHA
                    tests.append({
                        "hypothesis": label, "metric": metric, "baseline": base,
                        "mean_diff": float(d.mean()),
                        "ci95": [float(d.mean() - half), float(d.mean() + half)],
                        "test": r["test"], "p": float(r["p"]),
                        "p_corrected": float(r["p_corrected"]),
                        "shapiro_p": float(r["shapiro_p"]),
                        "primary_wins": f"{r['wins']}/{r['n']}",
                        "decision": "reject H0" if significant else "fail to reject H0",
                        "wrong_direction": bool(significant and d.mean() < 0),
                        "corrected_test_disagrees": bool((r["p_corrected"] < ALPHA) != significant),
                    })
            notes = ["Folds share training data; the corrected p-value is the conservative reference."]
            if n_pairs == 5: # If there are only 5 paired values, 
                # add a note about the limitations of the statistical tests due to the small sample size.
                notes.append("n=5: Wilcoxon cannot go below p=0.0625 and Shapiro-Wilk has little power.")
            report["splits"][split] = {"cv_run_files": n_files, "n_paired_values": n_pairs,
                                       "tests": tests, "notes": notes}
            log.event("stats", "tests computed", split=split, n_paired_values=n_pairs)
 
        #   Save the report as a JSON file in the "outputs" directory, along with CSV files for detailed metrics and connectivity information if available. 
        with open(os.path.join("outputs", "stats_results.json"), "w") as f:
            json.dump(report, f, indent=2)

        if all_rows: # Save the detailed metrics as a CSV file in the "outputs" directory, including columns for split, seed, fold, condition, metric, and value.
            with open(os.path.join("outputs", "tableau_metrics_long.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["split", "seed", "fold", "condition", "metric", "value"])
                w.writeheader()
                w.writerows(all_rows)
        if all_conn: # Save the connectivity information as a CSV file in the "outputs" directory, including columns for split, seed, fold, group, and any additional connectivity data.
            with open(os.path.join("outputs", "tableau_connectivity.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(all_conn[0].keys()))
                w.writeheader()
                w.writerows(all_conn)
        log.finish("ok", stats_file="outputs/stats_results.json")


if __name__ == "__main__":
    run()