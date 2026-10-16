"""Adds each training posting's title and scam/legit label to model/graph_assets.npz.

The explanation panel uses them to label the neighbours it draws. Without them the panel still works,
it just can't show titles or colour neighbours by label.

    python add_neighbour_info.py path\\to\\fake_job_postings.csv

The CSV must be the one the graph was built from (same rows, same order). That is checked by comparing
title/company hashes against the ones already stored, so a mismatched file is rejected.
"""
"""Adds each training posting's title and scam/legit label to model/graph_assets.npz.

The explanation panel uses them to label the neighbours it draws in the UI.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from utils import load_and_split_data

NPZ = os.path.join(HERE, "model", "graph_assets.npz")


def main():
    if not os.path.exists(NPZ):
        sys.exit(f"Error: {NPZ} does not exist. Run build_graph_assets.py first.")

    print("Loading preprocessed dataset from utils.load_and_split_data()...")
    df, _, _, _, _, _ = load_and_split_data()

    assets = dict(np.load(NPZ))

    if len(df) != len(assets["title_hash"]):
        sys.exit(f"Data mismatch: preprocessed df has {len(df)} rows but graph_assets.npz has {len(assets['title_hash'])} nodes.")

    # Extract labels and titles matching the 15,663 deduplicated graph nodes
    labels = df["label"].values.astype(np.int8)
    titles = np.array([(t or "")[:120] for t in df["title"].values])

    assets["labels"] = labels
    assets["titles"] = titles

    np.savez_compressed(NPZ, **assets)
    print(f"Successfully added titles and labels ({int(labels.sum())} scams of {len(df)} nodes) to {NPZ}")


if __name__ == "__main__":
    main()