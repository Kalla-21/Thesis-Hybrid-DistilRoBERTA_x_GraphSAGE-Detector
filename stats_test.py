# stats_test.py 
import glob, json, csv, os
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
TEST_TRAIN_RATIO = 0.29   


def load(split):
    files = sorted(glob.glob(f"outputs/cv_results_{split}_seed*.json"))
    res, conn, rows = {}, [], []
    for fp in files:
        with open(fp) as f:
            r = json.load(f)
        seed = r["config"]["seed"]
        for cond, folds in r["results"].items():
            if cond == "connectivity":
                for i, c in enumerate(folds):
                    for g, v in c.items():
                        conn.append(dict(split=split, seed=seed, fold=i + 1, group=g, **v))
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
    t, p_t = stats.ttest_rel(a, b)
    t_c = d.mean() / np.sqrt((1 / n + TEST_TRAIN_RATIO) * d.var(ddof=1))
    p_c = 2 * stats.t.sf(abs(t_c), n - 1)
    try:
        p_w = stats.wilcoxon(a, b)[1]
    except ValueError:
        p_w = float("nan")
    primary = "paired t" if sw_p > ALPHA else "Wilcoxon"
    p_prim = p_t if sw_p > ALPHA else p_w
    return dict(d=d, n=n, sw=sw_p, p_t=p_t, p_c=p_c, p_w=p_w, primary=primary, p_prim=p_prim,
                wins=int((d > 0).sum()))


def run():
    os.makedirs("outputs", exist_ok=True)
    all_rows, all_conn = [], []
    for split in ("random", "group"):
        res, n_files, rows, conn = load(split)
        all_rows += rows; all_conn += conn
        if not res:
            continue
        n_pairs = len(res[PRIMARY])
        print(f"\n##### {split.upper()} SPLIT ({n_files} CV run file(s), n={n_pairs} paired values) #####")
        if n_pairs == 5:
            print("  (n=5: Wilcoxon cannot go below p=0.0625; Shapiro has little power)")
        for label, base in HYPOTHESES:
            for m in METRICS:
                r = compare([x[m] for x in res[PRIMARY]], [x[m] for x in res[base]])
                d = r["d"]
                half = stats.t.ppf(0.975, r["n"] - 1) * d.std(ddof=1) / np.sqrt(r["n"])
                sig = r["p_prim"] < ALPHA
                verdict = "REJECT H0" if sig else "fail to reject"
                if sig and d.mean() < 0:
                    verdict += " (WRONG direction)"
                flag = "" if (r["p_c"] < ALPHA) == sig else "  <-- corrected test disagrees"
                print(f"{label:38s} {m.upper():3s} diff={d.mean():+.4f} [95% CI {d.mean()-half:+.4f},{d.mean()+half:+.4f}] "
                      f"{r['primary']} p={r['p_prim']:.4f} | corrected p={r['p_c']:.4f} | "
                      f"wins {r['wins']}/{r['n']} -> {verdict}{flag}")
    print("\nNote: folds share training data; corrected p is the conservative reference.")

    if all_rows:
        with open("outputs/tableau_metrics_long.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["split", "seed", "fold", "condition", "metric", "value"])
            w.writeheader(); w.writerows(all_rows)
    if all_conn:
        with open("outputs/tableau_connectivity.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_conn[0].keys()))
            w.writeheader(); w.writerows(all_conn)
        print("[INFO] wrote outputs/tableau_metrics_long.csv and outputs/tableau_connectivity.csv")


if __name__ == "__main__":
    run()