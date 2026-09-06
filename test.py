#test.py
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, matthews_corrcoef, confusion_matrix
)
from utils import DEVICE, GNN_DEVICE
from models.graphsage_model import FraudDetectionGraphSAGE

def evaluate():
    print("========== STARTING TEST EVALUATION SCRIPT ==========")
    print("[PROCESS] Loading air-gapped test environment (graph_data.pt)...")
    graph_data = torch.load("outputs/graph_data.pt", map_location=GNN_DEVICE, weights_only=False)
    graph_data = graph_data.to(GNN_DEVICE)

    print("[PROCESS] Initializing model architecture and loading best trained weights...")
    model = FraudDetectionGraphSAGE(
        in_channels=graph_data.x.shape[1], 
        tabular_dim=graph_data.x_tab.shape[1]
    ).to(GNN_DEVICE)
    model.load_state_dict(
        torch.load("outputs/graphsage_weights.pth", map_location=GNN_DEVICE, weights_only=True)
    )
    model.eval()

    threshold = 0.5 
    try:
        with open("outputs/best_threshold.json", "r") as f:
            threshold = json.load(f)["threshold"]
        print(f"[INFO] Loaded tuned decision threshold: {threshold:.3f}")
    except FileNotFoundError:
        print("[WARN] best_threshold.json not found — using default threshold 0.5")

    print("[PROCESS] Running final inference...")
    with torch.no_grad():
        # UPDATE: Add graph_data.x_tab to the forward pass
        logits = model(graph_data.x, graph_data.edge_index, graph_data.x_tab)
        probs  = torch.sigmoid(logits).squeeze()
        preds  = (probs >= threshold).float()

        y_true_train = graph_data.y[graph_data.train_mask].cpu().numpy()
        y_pred_train = preds[graph_data.train_mask].cpu().numpy()

        y_true_test = graph_data.y[graph_data.test_mask].cpu().numpy()
        y_pred_test = preds[graph_data.test_mask].cpu().numpy()

    def get_metrics(y_true, y_pred):
        return {
            "Accuracy":  accuracy_score(y_true, y_pred),
            "F1-Score":  f1_score(y_true, y_pred, zero_division=0),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall":    recall_score(y_true, y_pred, zero_division=0),
            "MCC":       matthews_corrcoef(y_true, y_pred),
        }

    print("[PROCESS] Calculating final metrics...")
    train_metrics = get_metrics(y_true_train, y_pred_train)
    test_metrics = get_metrics(y_true_test, y_pred_test)

    print("\n" + "=" * 45)
    print("FINAL SET RESULTS (TRAIN vs TEST)")
    print("=" * 45)
    print(f"{'Metric':<12} | {'Train Score':<12} | {'Test Score':<12}")
    print("-" * 45)
    for k in train_metrics.keys():
        print(f"{k:<12} | {train_metrics[k]:<12.4f} | {test_metrics[k]:<12.4f}")

    print("\n[PROCESS] Generating Confusion Matrix plot...")
    cm = confusion_matrix(y_true_test, y_pred_test)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Licit (0)", "Scam (1)"],
        yticklabels=["Licit (0)", "Scam (1)"],
    )
    plt.title(f"Test Set Confusion Matrix (threshold={threshold:.3f})")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    cm_path = "outputs/test_confusion_matrix.png"
    plt.savefig(cm_path)
    print(f"[INFO] Confusion Matrix saved to {cm_path}")

    print("\n--- TEST MATRIX BREAKDOWN ---")
    print(f"Correct Licit: {cm[0][0]:<5} | False Alarms: {cm[0][1]}")
    print(f"Missed Scams:  {cm[1][0]:<5} | Caught Scams: {cm[1][1]}")
    print("=====================================================")

if __name__ == "__main__":
    evaluate()