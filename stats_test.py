# stats_test.py
from logger import RunLogger

import csv
import glob
import json
import os

import numpy as np
from scipy import stats

PRIMARY = "cross_attention"
HYPOTHESES = [
    ("H1: hybrid vs DistilRoBERTa-only",   "distilroberta_only"),
    ("H2: hybrid vs GraphSAGE-only",       "graphsage_only"),
    ("H3: cross-attention vs late fusion", "late_concat"),
]
METRICS = ["f1", "mcc"]
ALPHA = 0.05
TEST_TRAIN_RATIO = 0.29   # n_test / n_train per fold (0.20 / 0.68), for the Nadeau-Bengio corrected t-test


def load(split):
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


def compare(a, b):
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


def run():
    os.makedirs("outputs", exist_ok=True)
    with RunLogger("stats_test") as log:
        all_rows, all_conn, report = [], [], {"alpha": ALPHA, "splits": {}}

        for split in ("random", "group"):
            res, n_files, rows, conn = load(split)
            all_rows += rows
            all_conn += conn
            if not res:
                log.event("stats", "no result files found", split=split)
                continue

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
            if n_pairs == 5:
                notes.append("n=5: Wilcoxon cannot go below p=0.0625 and Shapiro-Wilk has little power.")
            report["splits"][split] = {"cv_run_files": n_files, "n_paired_values": n_pairs,
                                       "tests": tests, "notes": notes}
            log.event("stats", "tests computed", split=split, n_paired_values=n_pairs)

        with open(os.path.join("outputs", "stats_results.json"), "w") as f:
            json.dump(report, f, indent=2)

        if all_rows:
            with open(os.path.join("outputs", "tableau_metrics_long.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["split", "seed", "fold", "condition", "metric", "value"])
                w.writeheader()
                w.writerows(all_rows)
        if all_conn:
            with open(os.path.join("outputs", "tableau_connectivity.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(all_conn[0].keys()))
                w.writeheader()
                w.writerows(all_conn)
        log.finish("ok", stats_file="outputs/stats_results.json")


if __name__ == "__main__":
    run()