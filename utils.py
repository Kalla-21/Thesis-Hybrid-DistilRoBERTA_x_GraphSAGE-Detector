# utils.py
import hashlib
import html
import os
import random
import re

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

try:
    import torch_directml
    DEVICE = torch_directml.device()
    GNN_DEVICE = torch.device("cpu")
    HARDWARE = {"backend": "directml", "encoder_device": str(DEVICE), "gnn_device": "cpu"}
except ImportError:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GNN_DEVICE = DEVICE
    HARDWARE = {"backend": DEVICE.type, "encoder_device": str(DEVICE), "gnn_device": str(GNN_DEVICE)}

# graph construction constants (stored inside evelynn.pt so inference builds edges the same way)
MAX_CLIQUE, K_NEIGHBORS, TITLE_CAP = 30, 10, 15

TEXT_FIELDS = ["title", "company_profile", "description", "requirements", "benefits"]
TABULAR_COLS = [
    "telecommuting", "has_company_logo", "has_questions",
    "has_company_profile", "has_requirements", "has_benefits",
    "has_valid_salary", "has_email", "has_phone", "has_url",
]  # the model also receives log1p(degree) as an 11th feature

_EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+')
_PHONE_RE = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Shared preprocessing helpers. Training (load_data) and inference (evelynn.py)
# both call these, so a new posting is processed exactly like the training data.
# ---------------------------------------------------------------------------
def clean_text(text):
    if pd.isna(text) or str(text).strip() == "":
        return ""
    text = html.unescape(str(text))
    text = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def normalize_title(title):
    return re.sub(r'[^a-z0-9\s]', '', str(title).lower().strip())


def find_emails(description):
    return _EMAIL_RE.findall(str(description).lower())


def find_phones(description):
    return _PHONE_RE.findall(str(description))


def valid_salary_flag(value):
    if pd.isna(value) or value == "" or "-" not in str(value):
        return 0.0
    try:
        parts = str(value).replace(',', '').split('-')
        low, high = float(parts[0]), float(parts[1])
    except ValueError:
        return 0.0
    return 0.0 if (low == 0 and high == 0) else 1.0


def build_full_text(title, company_profile, description, requirements, benefits):
    job_text = f"{title}. {description} {requirements} {benefits}".strip()
    return f"{job_text} {company_profile}"


def company_hash(profile):
    return hashlib.sha1(profile.encode("utf-8")).hexdigest()[:16]


def _num(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if v != v else v


def posting_to_features(posting):
    """Turn one raw posting (dict) into the model inputs: cleaned text, tabular flags, graph keys."""
    p = {field: clean_text(posting.get(field, "")) for field in TEXT_FIELDS}
    emails, phones = find_emails(p["description"]), find_phones(p["description"])
    flags = {
        "telecommuting": _num(posting.get("telecommuting")),
        "has_company_logo": _num(posting.get("has_company_logo")),
        "has_questions": _num(posting.get("has_questions")),
        "has_company_profile": float(p["company_profile"] != ""),
        "has_requirements": float(p["requirements"] != ""),
        "has_benefits": float(p["benefits"] != ""),
        "has_valid_salary": valid_salary_flag(posting.get("salary_range", "")),
        "has_email": float(len(emails) > 0),
        "has_phone": float(len(phones) > 0),
        "has_url": float("http" in p["description"].lower()),
    }
    return {
        **p,
        "full_text": build_full_text(p["title"], p["company_profile"], p["description"],
                                     p["requirements"], p["benefits"]),
        "norm_title": normalize_title(p["title"]),
        "contacts": emails + phones,
        "tab": [flags[c] for c in TABULAR_COLS],
    }


# ---------------------------------------------------------------------------
# Graph + dataset construction
# ---------------------------------------------------------------------------
def add_group_edges(indices, src, tgt, rng):
    """Fully connect small groups; link each member of a big group to K random members."""
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


def build_edges(df):
    """Edges between postings that share a company profile, a title (<= TITLE_CAP uses), or a contact."""
    rng = np.random.default_rng(42)
    src, tgt = [], []

    for company, group in df.groupby("company_profile"):
        if company != "":
            add_group_edges(group.index.tolist(), src, tgt, rng)

    title_counts = df["normalized_title"].value_counts()
    for title, group in df.groupby("normalized_title"):
        if title != "" and 1 < title_counts[title] <= TITLE_CAP:
            add_group_edges(group.index.tolist(), src, tgt, rng)

    contact_dict = {}
    for idx, emails, phones in zip(df.index, df["emails"], df["phones"]):
        for contact in emails + phones:
            contact_dict.setdefault(contact, []).append(idx)
    for indices in contact_dict.values():
        if len(indices) > 1:
            add_group_edges(indices, src, tgt, rng)

    return np.unique(np.array([src, tgt]), axis=1)


def _note(log, stage, message, **data):
    if log is not None:
        log.event(stage, message, **data)


def load_data(log=None, csv_path="data/fake_job_postings.csv"):
    """Returns (df, edges_arr, tabular_features). tabular_features has 11 columns (10 flags + log-degree)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}.")

    df = pd.read_csv(csv_path)
    rows_raw = len(df)

    # clean first, so postings that differ only by HTML markup count as duplicates
    for col in TEXT_FIELDS:
        df[col] = df[col].apply(clean_text)

    # drop exact duplicates; if labels conflict, the fraudulent copy is kept
    dedup_cols = ["title", "company_profile", "description"]
    dups = df[df.duplicated(subset=dedup_cols, keep=False)]
    conflicts = int((dups.groupby(dedup_cols)["fraudulent"].nunique() > 1).sum())
    df = df.sort_values(by="fraudulent", ascending=False)
    df = df.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
    _note(log, "data", "deduplicated", rows_raw=rows_raw, rows=len(df),
          conflicting_label_groups=conflicts, fraud=int(df["fraudulent"].sum()))

    df["has_company_profile"] = (df["company_profile"] != "").astype(float)
    df["has_requirements"] = (df["requirements"] != "").astype(float)
    df["has_benefits"] = (df["benefits"] != "").astype(float)
    df.fillna("", inplace=True)

    df["full_text"] = [
        build_full_text(t, c, d, r, b)
        for t, c, d, r, b in zip(df["title"], df["company_profile"], df["description"],
                                 df["requirements"], df["benefits"])
    ]
    df["label"] = df["fraudulent"].astype(int)

    # one group per company profile (postings without a profile are their own group); used by the group split
    company_key = df["company_profile"].where(df["company_profile"] != "", "solo_" + df.index.astype(str))
    df["group_id"] = pd.factorize(company_key)[0]

    df["has_valid_salary"] = df["salary_range"].apply(valid_salary_flag)
    df["emails"] = df["description"].apply(find_emails)
    df["phones"] = df["description"].apply(find_phones)
    df["has_email"] = df["emails"].apply(lambda x: float(len(x) > 0))
    df["has_phone"] = df["phones"].apply(lambda x: float(len(x) > 0))
    df["has_url"] = df["description"].apply(lambda x: float("http" in x.lower()))
    for col in TABULAR_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
    tabular_features = df[TABULAR_COLS].values

    df["normalized_title"] = df["title"].apply(normalize_title)
    edges_arr = build_edges(df)

    deg = np.bincount(edges_arr[0], minlength=len(df))
    tabular_features = np.column_stack([tabular_features, np.log1p(deg)])

    _note(log, "data", "graph built", directed_edges=int(edges_arr.shape[1]),
          isolated_postings=int((deg == 0).sum()), tabular_dim=int(tabular_features.shape[1]))
    return df, edges_arr, tabular_features


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def _mask(n, idx):
    m = torch.zeros(n, dtype=torch.bool)
    m[idx] = True
    return m


def iid_masks(labels, seed=42):
    """Stratified 70/15/15 train/val/test masks (used when building the final model)."""
    idx = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(idx, test_size=0.3, stratify=labels, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=seed)
    return [_mask(len(labels), i) for i in (train_idx, val_idx, test_idx)]


def stratified_kfold_masks(labels, k=5, seed=42, groups=None):
    """Yields (fold, train_mask, val_mask, test_mask). With `groups`, no company appears on both sides."""
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
        yield fold_i, _mask(n, train_idx), _mask(n, val_idx), _mask(n, test_idx)


def connectivity_breakdown(probs, thresh, y, test_mask, edge_index):
    """Recall and false alarms on test postings that have neighbours vs. those that have none."""
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