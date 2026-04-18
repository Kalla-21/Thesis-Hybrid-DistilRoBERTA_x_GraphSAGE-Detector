import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    matthews_corrcoef, confusion_matrix
)
from torch.utils.data import Dataset

try:
    import torch_directml
    DEVICE = torch_directml.device()
    GNN_DEVICE = torch.device("cpu")  # DirectML lacks PyG index_select support
    print(f"[SYSTEM] Hardware detected: DirectML ({DEVICE}), GNN will run on CPU")
except ImportError:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GNN_DEVICE = DEVICE
    print(f"[SYSTEM] Hardware detected: {DEVICE}")

class JobTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }

def load_and_split_data(csv_path="data/fake_job_postings.csv"):
    print("\n--- STAGE 1: DATA INGESTION ---")
    print(f"[PROCESS] Loading data from {csv_path}...")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] Could not find {csv_path}.")

    df = pd.read_csv(csv_path)

    print("[PROCESS] Preprocessing text and mapping labels...")
    df.fillna("", inplace=True)

    df["full_text"] = (
        df["title"] + " " +
        df["company_profile"] + " " +
        df["description"] + " " +
        df["requirements"] + " " +
        df["benefits"]
    )
    df["label"] = df["fraudulent"].astype(int)

    # --- NEW: Extract Tabular Metadata ---
    print("[PROCESS] Extracting binary/tabular features...")
    tabular_cols = ["telecommuting", "has_company_logo", "has_questions"]
    for col in tabular_cols:
        # Convert to float for neural network compatibility
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)
    tabular_features = df[tabular_cols].values

    # --- EXPANDED: Graph Edges ---
    print("[PROCESS] Building expanded graph edges...")
    edges_source, edges_target = [], []

    # 1. Edges based on Company Profile (Your original logic)
    for company, group in df.groupby("company_profile"):
        if company != "":
            indices = group.index.tolist()
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    edges_source.extend([indices[i], indices[j]])
                    edges_target.extend([indices[j], indices[i]])

# 2. Edges based on identical Titles
    title_counts = df['title'].value_counts()
    for title, group in df.groupby("title"):
        # Lowered from 100 to 15 to prevent edge explosions on generic titles
        if title != "" and 1 < title_counts[title] <= 15: 
            indices = group.index.tolist()
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    edges_source.extend([indices[i], indices[j]])
                    edges_target.extend([indices[j], indices[i]])
                    
    edges_arr = np.array([edges_source, edges_target])
    # Remove any duplicate edges created by jobs sharing both company and title
    edges_arr = np.unique(edges_arr, axis=1)
    
    print(f"[INFO] Generated {edges_arr.shape[1]} unique edges.")

    print("[PROCESS] Performing IID stratified split (70/15/15)...")
    all_indices = np.arange(len(df))
    labels_arr = df["label"].values

    train_idx, temp_idx = train_test_split(all_indices, test_size=0.3, stratify=labels_arr, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=labels_arr[temp_idx], random_state=42)

    print("[PROCESS] Generating PyTorch boolean masks...")
    train_mask = torch.zeros(len(df), dtype=torch.bool)
    val_mask   = torch.zeros(len(df), dtype=torch.bool)
    test_mask  = torch.zeros(len(df), dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    print("[INFO] Stage 1 Complete.\n")
    # Return tabular_features at the end
    return df, edges_arr, train_mask, val_mask, test_mask, tabular_features

def plot_training_history(history, model_name):
    print(f"\n[PROCESS] Generating training plots for {model_name}...")
    os.makedirs("outputs", exist_ok=True)
    epochs_range = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs_range, history["loss"], "r-o", label="Training Loss")
    axes[0].set_title(f"{model_name}: Loss")
    axes[0].legend()

    axes[1].plot(epochs_range, history["f1"],     "b-s", label="F1-Score")
    axes[1].plot(epochs_range, history["recall"], "g-d", label="Recall")
    axes[1].set_title(f"{model_name}: Validation Metrics")
    axes[1].legend()

    plt.tight_layout()
    save_path = f"outputs/{model_name.lower().replace(' ', '_')}_history.png"
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Plot saved to {save_path}")