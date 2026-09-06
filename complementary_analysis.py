#complementary_analysis.py
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils import DEVICE, GNN_DEVICE, JobTextDataset, set_seed, load_and_split_data
from models.distilroberta_encoder import DistilRobertaModule
from models.graphsage_model import (
    FraudDetectionGraphSAGE, DistilRobertaOnlyClassifier, GraphSAGEOnlyBaseline,
)
from train_cv import train_one_model, CV_CONFIG

SEED = 42


def run():
    set_seed(SEED)
    os.makedirs("outputs", exist_ok=True)

    df, edges_arr, train_mask, val_mask, test_mask, tabular_features = load_and_split_data()
    labels_arr = df["label"].values

    emb_path = "outputs/embeddings.npy"
    if os.path.exists(emb_path):
        print("[INFO] Loading cached text embeddings...")
        embeddings = np.load(emb_path)
    else:
        text_module = DistilRobertaModule(DEVICE)
        full_dataset = JobTextDataset(df["full_text"].values, df["label"].values, text_module.tokenizer)
        full_loader = DataLoader(full_dataset, batch_size=16, shuffle=False)
        embeddings = text_module.extract_embeddings(full_loader)
        np.save(emb_path, embeddings)

    x_text = torch.tensor(embeddings, dtype=torch.float).to(GNN_DEVICE)
    x_tab = torch.tensor(tabular_features, dtype=torch.float).to(GNN_DEVICE)
    y = torch.tensor(labels_arr, dtype=torch.float).to(GNN_DEVICE)
    edge_index = torch.tensor(edges_arr, dtype=torch.long).to(GNN_DEVICE)
    num_nodes = x_text.shape[0]

    train_mask = train_mask.to(GNN_DEVICE)
    val_mask = val_mask.to(GNN_DEVICE)
    test_mask = test_mask.to(GNN_DEVICE)

    num_licit = (y[train_mask] == 0).sum().item()
    num_scam = (y[train_mask] == 1).sum().item()
    pos_weight = torch.tensor(
        [min(num_licit / max(num_scam, 1), CV_CONFIG["pos_weight_cap"])], dtype=torch.float
    ).to(GNN_DEVICE)

    preds = {}

    print("[PROCESS] Training DistilRoBERTa-only...")
    set_seed(SEED)
    model = DistilRobertaOnlyClassifier(text_dim=x_text.shape[1])
    _, thresh, probs = train_one_model(
        model, lambda m: m(x_text), y, train_mask, val_mask, test_mask,
        pos_weight, CV_CONFIG, GNN_DEVICE,
    )
    preds["distilroberta_only"] = (probs >= thresh).astype(int)

    print("[PROCESS] Training GraphSAGE-only...")
    set_seed(SEED)
    model = GraphSAGEOnlyBaseline(num_nodes=num_nodes, rand_dim=x_text.shape[1], seed=SEED)
    _, thresh, probs = train_one_model(
        model, lambda m: m(edge_index), y, train_mask, val_mask, test_mask,
        pos_weight, CV_CONFIG, GNN_DEVICE,
    )
    preds["graphsage_only"] = (probs >= thresh).astype(int)

    print("[PROCESS] Training cross-attention hybrid...")
    set_seed(SEED)
    model = FraudDetectionGraphSAGE(
        text_dim=x_text.shape[1], tabular_dim=x_tab.shape[1], fusion_mode="cross_attention"
    )
    _, thresh, probs = train_one_model(
        model, lambda m: m(x_text, edge_index, x_tab), y, train_mask, val_mask, test_mask,
        pos_weight, CV_CONFIG, GNN_DEVICE,
    )
    preds["cross_attention"] = (probs >= thresh).astype(int)

    # --- Restrict to actually-fraudulent test listings ---
    test_idx = test_mask.cpu().numpy().nonzero()[0]
    y_np = y.cpu().numpy()
    fraud_idx = test_idx[y_np[test_idx] == 1]
    print(f"\n[INFO] {len(fraud_idx)} fraudulent listings in the test set.")

    text_hit = preds["distilroberta_only"][fraud_idx] == 1
    graph_hit = preds["graphsage_only"][fraud_idx] == 1
    hybrid_hit = preds["cross_attention"][fraud_idx] == 1

    neither = ~text_hit & ~graph_hit
    text_only = text_hit & ~graph_hit
    graph_only = ~text_hit & graph_hit
    both = text_hit & graph_hit

    report = {
        "n_fraud_in_test": int(len(fraud_idx)),
        "caught_by_neither_standalone": int(neither.sum()),
        "caught_by_text_only": int(text_only.sum()),
        "caught_by_graph_only": int(graph_only.sum()),
        "caught_by_both_standalone": int(both.sum()),
        "of_text_misses_graph_catches": int((~text_hit & graph_hit).sum()),
        "of_graph_misses_text_catches": int((~graph_hit & text_hit).sum()),
        "hybrid_recovers_of_neither_caught": {
            "denominator": int(neither.sum()),
            "hybrid_catches": int((hybrid_hit & neither).sum()),
        },
        "hybrid_overall_recall_on_fraud": float(hybrid_hit.mean()),
        "distilroberta_only_recall_on_fraud": float(text_hit.mean()),
        "graphsage_only_recall_on_fraud": float(graph_hit.mean()),
    }

    with open("outputs/complementarity_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n===== COMPLEMENTARITY BREAKDOWN (fraudulent test listings only) =====")
    print(f"Caught by neither standalone model : {report['caught_by_neither_standalone']}")
    print(f"Caught by DistilRoBERTa only        : {report['caught_by_text_only']}")
    print(f"Caught by GraphSAGE only             : {report['caught_by_graph_only']}")
    print(f"Caught by both standalone models     : {report['caught_by_both_standalone']}")
    n_neither = report["hybrid_recovers_of_neither_caught"]["denominator"]
    n_recovered = report["hybrid_recovers_of_neither_caught"]["hybrid_catches"]
    if n_neither > 0:
        print(f"\nOf the {n_neither} cases BOTH standalone models missed, "
              f"the hybrid catches {n_recovered} "
              f"({100 * n_recovered / n_neither:.1f}%).")
    else:
        print("\nNo cases were missed by both standalone models in this split -- "
              "re-check on a split/fold with more test-set fraud cases, or run "
              "this analysis per-fold across the CV splits for a sturdier count.")
    print(f"\nRecall on fraud -- DistilRoBERTa-only: {report['distilroberta_only_recall_on_fraud']:.3f} | "
          f"GraphSAGE-only: {report['graphsage_only_recall_on_fraud']:.3f} | "
          f"Hybrid: {report['hybrid_overall_recall_on_fraud']:.3f}")

    # --- Simple bar chart of the breakdown ---
    labels = ["Neither\nstandalone", "Text\nonly", "Graph\nonly", "Both\nstandalone"]
    counts = [neither.sum(), text_only.sum(), graph_only.sum(), both.sum()]
    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, counts, color=["#c0392b", "#2980b9", "#27ae60", "#8e44ad"])
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                  str(int(count)), ha="center")
    plt.ylabel("Number of fraudulent test listings")
    plt.title("Which model catches which fraud cases\n(standalone DistilRoBERTa vs. standalone GraphSAGE)")
    plt.tight_layout()
    plt.savefig("outputs/complementarity_breakdown.png")
    plt.close()
    print("\n[INFO] Saved outputs/complementarity_breakdown.png")


if __name__ == "__main__":
    run()