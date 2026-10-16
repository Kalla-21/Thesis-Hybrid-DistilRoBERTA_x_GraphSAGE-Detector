# utils.py
import os
import random
import re
import html
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from torch.utils.data import Dataset

try:
    import torch_directml
    DEVICE = torch_directml.device()
    GNN_DEVICE = torch.device("cpu")
    print(f"[SYSTEM] Hardware detected: DirectML ({DEVICE}), GNN will run on CPU")
except ImportError:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GNN_DEVICE = DEVICE
    print(f"[SYSTEM] Hardware detected: {DEVICE}")

MAX_CLIQUE, K_NEIGHBORS = 30, 10


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


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[SYSTEM] Random seed set to {seed}")


def add_group_edges(indices, src, tgt, rng):
    """Fully connect small groups; link each member of big groups to K random members."""
    n = len(indices)
    if n < 2:
        return
    if n <= MAX_CLIQUE:
        pairs = [(indices[i], indices[j]) for i in range(n) for j in range(i + 1, n)]
    else:
        pairs = []
        for a in indices:
            for b in rng.choice(indices, size=K_NEIGHBORS, replace=False):
                if a != b:
                    pairs.append((a, b))
    for a, b in pairs:
        src.extend([a, b])
        tgt.extend([b, a])


def load_and_split_data(csv_path="data/fake_job_postings.csv"):
    print("\n--- STAGE 1: DATA INGESTION ---")
    print(f"[PROCESS] Loading data from {csv_path}...")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] Could not find {csv_path}.")

    df = pd.read_csv(csv_path)

    # 1. clean text FIRST so HTML-only variants count as duplicates
    def clean_text(text):
        if pd.isna(text) or str(text).strip() == "":
            return ""
        text = html.unescape(str(text))
        text = re.sub(r'<[^>]+>', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    text_cols = ["title", "company_profile", "description", "requirements", "benefits"]
    print("[PROCESS] Preprocessing text and scrubbing HTML tags...")
    for col in text_cols:
        df[col] = df[col].apply(clean_text)

    # 2. deduplicate on cleaned text; keep the fraudulent copy if labels conflict
    dedup_cols = ['title', 'company_profile', 'description']
    dups = df[df.duplicated(subset=dedup_cols, keep=False)]
    conflicts = dups.groupby(dedup_cols)['fraudulent'].nunique()
    print(f"[INFO] Found {(conflicts > 1).sum()} unique listings with conflicting labels.")
    df = df.sort_values(by='fraudulent', ascending=False)
    df = df.drop_duplicates(subset=dedup_cols, keep='first').reset_index(drop=True)
    print(f"[INFO] Dataset reduced to {len(df)} rows after exact-text deduplication.")

    # 3. missing flags
    df['has_company_profile'] = df['company_profile'].apply(lambda x: 1.0 if x != "" else 0.0)
    df['has_requirements'] = df['requirements'].apply(lambda x: 1.0 if x != "" else 0.0)
    df['has_benefits'] = df['benefits'].apply(lambda x: 1.0 if x != "" else 0.0)

    df.fillna("", inplace=True)

    # 4. text fields: job text and company profile are embedded separately
    df["job_text"] = (df["title"] + ". " + df["description"] + " " +
                      df["requirements"] + " " + df["benefits"]).str.strip()
    df["full_text"] = df["job_text"] + " " + df["company_profile"]
    df["label"] = df["fraudulent"].astype(int)

    # 5. group ids for company-level splitting
    company_key = df["company_profile"].where(
        df["company_profile"] != "", "solo_" + df.index.astype(str)
    )
    df["group_id"] = pd.factorize(company_key)[0]

    # 6. salary
    def parse_salary(val):
        if pd.isna(val) or val == "" or "-" not in str(val):
            return 0.0, 0.0, 0.0
        try:
            parts = str(val).replace(',', '').split('-')
            s_min, s_max = float(parts[0]), float(parts[1])
            if s_min == 0 and s_max == 0:
                return 0.0, 0.0, 0.0
            return s_min, s_max, 1.0
        except ValueError:
            return 0.0, 0.0, 0.0

    salary_data = df['salary_range'].apply(parse_salary).tolist()
    df['salary_min'] = [x[0] for x in salary_data]
    df['salary_max'] = [x[1] for x in salary_data]
    df['has_valid_salary'] = [x[2] for x in salary_data]

    # 7. contact features
    df['emails'] = df['description'].apply(lambda x: re.findall(r'[\w\.-]+@[\w\.-]+', str(x).lower()))
    df['phones'] = df['description'].apply(lambda x: re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', str(x)))
    df['has_email'] = df['emails'].apply(lambda x: 1.0 if len(x) > 0 else 0.0)
    df['has_phone'] = df['phones'].apply(lambda x: 1.0 if len(x) > 0 else 0.0)
    df['has_url'] = df['description'].apply(lambda x: 1.0 if 'http' in str(x).lower() else 0.0)

    print("[PROCESS] Extracting binary/tabular features...")
    tabular_cols = [
        "telecommuting", "has_company_logo", "has_questions",
        "has_company_profile", "has_requirements", "has_benefits",
        "has_valid_salary", "has_email", "has_phone", "has_url",
    ]
    for col in tabular_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)
    tabular_features = df[tabular_cols].values

    # 8. graph edges
    df['normalized_title'] = df['title'].str.lower().str.strip()
    df['normalized_title'] = df['normalized_title'].str.replace(r'[^a-z0-9\s]', '', regex=True)

    rng = np.random.default_rng(42)
    edges_source, edges_target = [], []

    for company, group in df.groupby("company_profile"):
        if company != "":
            add_group_edges(group.index.tolist(), edges_source, edges_target, rng)

    title_counts = df['normalized_title'].value_counts()
    for title, group in df.groupby("normalized_title"):
        if title != "" and 1 < title_counts[title] <= 15:
            add_group_edges(group.index.tolist(), edges_source, edges_target, rng)

    print("[PROCESS] Mapping identity rings (shared emails/phones)...")
    contact_dict = {}
    for idx, row in df.iterrows():
        for contact in row['emails'] + row['phones']:
            contact_dict.setdefault(contact, []).append(idx)
    for contact, indices in contact_dict.items():
        if len(indices) > 1:
            add_group_edges(indices, edges_source, edges_target, rng)

    edges_arr = np.unique(np.array([edges_source, edges_target]), axis=1)
    print(f"[INFO] Generated {edges_arr.shape[1]} unique edges.")

    # 9. degree as an extra tabular feature
    deg = np.bincount(edges_arr[0], minlength=len(df))
    tabular_features = np.column_stack([tabular_features, np.log1p(deg)])

    # 10. default IID split (train_cv.py builds its own folds)
    print("[PROCESS] Performing IID stratified split (70/15/15)...")
    all_indices = np.arange(len(df))
    labels_arr = df["label"].values
    train_idx, temp_idx = train_test_split(all_indices, test_size=0.3, stratify=labels_arr, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=labels_arr[temp_idx], random_state=42)

    train_mask = torch.zeros(len(df), dtype=torch.bool)
    val_mask = torch.zeros(len(df), dtype=torch.bool)
    test_mask = torch.zeros(len(df), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    print("[INFO] Stage 1 Complete.\n")
    return df, edges_arr, train_mask, val_mask, test_mask, tabular_features


def plot_training_history(history, model_name):
    print(f"\n[PROCESS] Generating training plots for {model_name}...")
    os.makedirs("outputs", exist_ok=True)
    epochs_range = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs_range, history["loss"], "r-o", label="Training Loss")
    axes[0].set_title(f"{model_name}: Loss")
    axes[0].legend()

    axes[1].plot(epochs_range, history["f1"], "b-s", label="F1-Score")
    axes[1].plot(epochs_range, history["recall"], "g-d", label="Recall")
    axes[1].set_title(f"{model_name}: Validation Metrics")
    axes[1].legend()

    plt.tight_layout()
    save_path = f"outputs/{model_name.lower().replace(' ', '_')}_history.png"
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Plot saved to {save_path}")


def stratified_kfold_masks(labels, k=5, seed=42, groups=None):
    n = len(labels)
    if groups is None:
        outer = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(n), labels)
    else:
        outer = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(n), labels, groups)
    for fold_i, (train_idx, test_idx) in enumerate(outer):
        if groups is None:
            train_idx, val_idx = train_test_split(
                train_idx, test_size=0.15, stratify=labels[train_idx], random_state=seed
            )
        else:
            inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
            tr, va = next(inner.split(np.zeros(len(train_idx)), labels[train_idx], groups[train_idx]))
            train_idx, val_idx = train_idx[tr], train_idx[va]
        masks = []
        for idx in (train_idx, val_idx, test_idx):
            m = torch.zeros(n, dtype=torch.bool)
            m[idx] = True
            masks.append(m)
        yield fold_i, *masks


def connectivity_breakdown(probs, thresh, y, test_mask, edge_index):
    n = len(y)
    deg = torch.bincount(edge_index[0], minlength=n).cpu().numpy()
    y_np, tm = y.cpu().numpy(), test_mask.cpu().numpy()
    pred = (probs >= thresh).astype(int)
    out = {}
    for name, grp in [("connected", deg > 0), ("isolated", deg == 0)]:
        idx = tm & grp
        pos, neg = idx & (y_np == 1), idx & (y_np == 0)
        out[name] = {
            "n_test": int(idx.sum()),
            "n_scams": int(pos.sum()),
            "scams_caught": int(pred[pos].sum()),
            "recall": float(pred[pos].mean()) if pos.sum() else None,
            "false_alarms": int(pred[neg].sum()),
            "n_legit": int(neg.sum()),
        }
    return out