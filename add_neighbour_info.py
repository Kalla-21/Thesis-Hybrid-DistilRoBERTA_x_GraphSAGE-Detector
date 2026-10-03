"""Adds each training posting's title and scam/legit label to model/graph_assets.npz.

The explanation panel uses them to label the neighbours it draws. Without them the panel still works,
it just can't show titles or colour neighbours by label.

    python add_neighbour_info.py path\\to\\fake_job_postings.csv

The CSV must be the one the graph was built from (same rows, same order). That is checked by comparing
title/company hashes against the ones already stored, so a mismatched file is rejected.
"""
import csv
import hashlib
import os
import sys

import numpy as np

NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "graph_assets.npz")


def h64(s: str) -> int:
    if s == "":
        return 0
    return int(np.frombuffer(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), dtype=np.uint64)[0])


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    with open(sys.argv[1], encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assets = dict(np.load(NPZ))
    if len(rows) != len(assets["title_hash"]):
        sys.exit(f"CSV has {len(rows)} rows but the graph has {len(assets['title_hash'])} nodes - wrong file.")
    title_hash = np.array([h64(r["title"] or "") for r in rows], dtype=np.uint64)
    company_hash = np.array([h64(r["company_profile"] or "") for r in rows], dtype=np.uint64)
    if not (title_hash == assets["title_hash"]).all() or not (company_hash == assets["company_hash"]).all():
        sys.exit("CSV rows don't line up with the saved graph (title/company hashes differ) - wrong file or version.")

    assets["labels"] = np.array([int(r["fraudulent"]) for r in rows], dtype=np.int8)
    assets["titles"] = np.array([(r["title"] or "")[:120] for r in rows])
    np.savez_compressed(NPZ, **assets)
    print(f"Added titles and labels ({int(assets['labels'].sum())} scams of {len(rows)}) to {NPZ}")


if __name__ == "__main__":
    main()
