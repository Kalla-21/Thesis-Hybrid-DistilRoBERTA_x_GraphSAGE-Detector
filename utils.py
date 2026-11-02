# utils.py
# preprocessing, graph construction, and train/val/test splits for the Evelynn model.
import hashlib
import html
import os
import random
import re
from shutil import which

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

try: # for non-CUDA Windows users, this will use DirectML as the backend for PyTorch. If DirectML is not available, it will fall back to CUDA if available, or CPU otherwise.
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


def set_seed(seed=42): # sets the random seed for reproducibility across different libraries and frameworks. 
    # It ensures that the results of experiments are consistent and can be reproduced by others. Ehem, perhaps future researchers. :3 
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
def clean_text(text): # returns an empty string if the input is None or empty, 
    #otherwise it unescapes HTML entities, removes HTML tags, and collapses whitespace to a single space.
    if pd.isna(text) or str(text).strip() == "":
        return ""
    text = html.unescape(str(text))
    text = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def normalize_title(title): # returns a lowercase, stripped version of the input title with all non-alphanumeric characters removed.
    # 2 examples: "Senior Software Engineer (Python)" -> "senior software engineer python" " Sr. Developer (Remote) -> sr developer remote"
    return re.sub(r'[^a-z0-9\s]', '', str(title).lower().strip())


def find_emails(description): # lowercases the input description and returns a list of all email addresses found in it using a regular expression pattern.
    return _EMAIL_RE.findall(str(description).lower())


def find_phones(description): # returns a list of all phone numbers found in the input description using a regular expression pattern that 
    #matches 10-digit phone numbers with optional separators (hyphens or dots).
    return _PHONE_RE.findall(str(description))


def valid_salary_flag(value): # returns 0.0 if the input value is None, empty, or does not contain a hyphen. 
    # Otherwise, it attempts to parse the value as a salary range (e.g., "50,000-70,000") and returns 1.0 if the range is valid 
    # (i.e., both low and high values are non-zero), or 0.0 if the range is invalid (i.e., both low and high values are zero).
    if pd.isna(value) or value == "" or "-" not in str(value):
        return 0.0
    try:
        parts = str(value).replace(',', '').split('-')
        low, high = float(parts[0]), float(parts[1])
    except ValueError:
        return 0.0
    return 0.0 if (low == 0 and high == 0) else 1.0


def build_full_text(title, company_profile, description, requirements, benefits): # returns a string that concatenates the input title, description, requirements, benefits,
    # and company profile into a single text string.
    job_text = f"{title}. {description} {requirements} {benefits}".strip()
    return f"{job_text} {company_profile}" # make sure the company profile is always at the end, so that it is not truncated by the tokenizer if the text is too long.


def company_hash(profile): # returns a 16-character hexadecimal SHA-1 hash of the input company profile string.
    return hashlib.sha1(profile.encode("utf-8")).hexdigest()[:16]


def _num(value): # all other values with NaN or null are converted to 0.0, and all other values are converted to float. If the conversion fails, it returns 0.0.
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if v != v else v


def posting_to_features(posting):
    # returns a dict with the same keys as the training data, including "full_text", "norm_title", "contacts", and "tab" (the 11 tabular features).
    # key-value pairs are extracted from the input posting dict, and the values are cleaned and processed using the helper functions defined above.
    # 1 cleans the text fields, 2 finds emails and phones in the description, 3 builds the full text in order, 
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
    return { #4 normalizes the title, 5 creates a list of contacts, and 6 creates a list of tabular features based on the flags defined above.
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
    # Adds edges between all pairs of indices in the input list, with a maximum of K_NEIGHBORS edges per node if the group size exceeds MAX_CLIQUE.
    n = len(indices)
    if n < 2:
        return # do nothing with groups that have fewer than 2 postings, since there are no edges to add.
    if n <= MAX_CLIQUE: # for every pair (i, j) pair with i < j.
        pairs = [(indices[i], indices[j]) for i in range(n) for j in range(i + 1, n)]
    else:
        pairs = [] # more than 30 posings share the same company profile, title, or contact. 
        # Randomly select K_NEIGHBORS neighbors for each posting to limit the number of edges.
        for a in indices:
            for b in rng.choice(indices, size=K_NEIGHBORS, replace=False):
                if a != b:
                    pairs.append((a, b))
    for a, b in pairs: # add edges in both directions to create an UNDIRECTED graph. 
        # The source node is a and the target node is b, and vice versa.
        src.extend([a, b])
        tgt.extend([b, a])


def build_edges(df):
    # Builds the graph edges based on company profile, normalized title, and contacts (emails and phones) in the input dataframe.
    rng = np.random.default_rng(42)
    src, tgt = [], []

    # first rule in edge building: company profile. All postings with the same non-empty company profile are connected to each other.
    for company, group in df.groupby("company_profile"):
        if company != "":
            add_group_edges(group.index.tolist(), src, tgt, rng)

    # second rule in edge building: normalized title. All postings with the same non-empty normalized title are connected to each other, 
    # but only if the number of postings with that title is between 2 and TITLE_CAP (15).
    title_counts = df["normalized_title"].value_counts()
    for title, group in df.groupby("normalized_title"):
        if title != "" and 1 < title_counts[title] <= TITLE_CAP:
            add_group_edges(group.index.tolist(), src, tgt, rng)

    # third rule in edge building: shared contacts. All postings with the same email or phone number are connected to each other.
    contact_dict = {}
    for idx, emails, phones in zip(df.index, df["emails"], df["phones"]):
        for contact in emails + phones:
            contact_dict.setdefault(contact, []).append(idx)
    for indices in contact_dict.values():
        if len(indices) > 1:
            add_group_edges(indices, src, tgt, rng)

    return np.unique(np.array([src, tgt]), axis=1) #use unique to remove duplicate edges, 2xE array of edges, where E is the number of unique edges. 
                                    # The first row contains the source nodes and the second row contains the target nodes.
          

def _note(log, stage, message, **data):
    # Logs a message with the given stage and message, along with any additional data provided as keyword arguments.
    if log is not None:
        log.event(stage, message, **data)


def load_data(log=None, csv_path="data/fake_job_postings.csv"):
    # Loads the dataset from the specified CSV file, preprocesses the data, and constructs the graph edges and tabular features.
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}.")

    df = pd.read_csv(csv_path) # change this to read the CSV file into a pandas DataFrame, 
    # which is a tabular data structure that allows for easy manipulation and analysis of the data.
    rows_raw = len(df)

    # clean first, so postings that differ only by HTML markup count as duplicates
    for col in TEXT_FIELDS:
        df[col] = df[col].apply(clean_text)

    # drop exact duplicates; if labels conflict, the fraudulent copy is kept
    dedup_cols = ["title", "company_profile", "description"] 
    dups = df[df.duplicated(subset=dedup_cols, keep=False)] 
    conflicts = int((dups.groupby(dedup_cols)["fraudulent"].nunique() > 1).sum()) # keep fraudulent copy if labels conflict, 
                                                        # and count the number of conflicting label groups.
    df = df.sort_values(by="fraudulent", ascending=False)
    df = df.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
    _note(log, "data", "deduplicated", rows_raw=rows_raw, rows=len(df),
          conflicting_label_groups=conflicts, fraud=int(df["fraudulent"].sum()))

    df["has_company_profile"] = (df["company_profile"] != "").astype(float)
    df["has_requirements"] = (df["requirements"] != "").astype(float)
    df["has_benefits"] = (df["benefits"] != "").astype(float)
    df.fillna("", inplace=True) # ^^ fill any remaining NaN values in the DataFrame with empty strings, 
    # ensuring that all text fields are non-null and can be processed without errors.

    df["full_text"] = [
        build_full_text(t, c, d, r, b) # uses the build_full_text function to create a new column "full_text" in the DataFrame,
        for t, c, d, r, b in zip(df["title"], df["company_profile"], df["description"],
                                 df["requirements"], df["benefits"])
    ]
    df["label"] = df["fraudulent"].astype(int)

    # one group per company profile (postings without a profile are their own group); used by the group split
    company_key = df["company_profile"].where(df["company_profile"] != "", "solo_" + df.index.astype(str))
    df["group_id"] = pd.factorize(company_key)[0] # creates a new column "group_id" in the DataFrame, which assigns a unique integer ID to each company profile.

    # helper columns for graph construction
    df["has_valid_salary"] = df["salary_range"].apply(valid_salary_flag)
    df["emails"] = df["description"].apply(find_emails)
    df["phones"] = df["description"].apply(find_phones)
    df["has_email"] = df["emails"].apply(lambda x: float(len(x) > 0))
    df["has_phone"] = df["phones"].apply(lambda x: float(len(x) > 0))
    df["has_url"] = df["description"].apply(lambda x: float("http" in x.lower()))
    
    # build the graph edges and tabular features, non-numerical is set to 0.0, and the degree of each node is added as an additional feature.
    for col in TABULAR_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
    tabular_features = df[TABULAR_COLS].values

    # normalize the title and build the graph edges based on company profile, normalized title, and contacts (emails and phones).
    df["normalized_title"] = df["title"].apply(normalize_title)
    edges_arr = build_edges(df) # make the graph undirected by adding edges in both directions, 
    #and remove duplicate edges. The resulting edges are stored in a 2xE array, where E is the number of unique edges.

    # calculate the degree of each node in the graph and add it as an additional feature to the tabular features.
    # this is the weights of the nodes in the graph, which is used to capture the connectivity of each posting in the graph.
    deg = np.bincount(edges_arr[0], minlength=len(df))
    tabular_features = np.column_stack([tabular_features, np.log1p(deg)])

    # log the number of directed edges, isolated postings, and the dimension of the tabular features.
    _note(log, "data", "graph built", directed_edges=int(edges_arr.shape[1]),
          isolated_postings=int((deg == 0).sum()), tabular_dim=int(tabular_features.shape[1]))
    return df, edges_arr, tabular_features


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def _mask(n, idx): # returns a boolean mask of length n, where the indices specified in idx are set to True and all other indices are set to False.
    m = torch.zeros(n, dtype=torch.bool)
    m[idx] = True
    return m


def iid_masks(labels, seed=42): # stratified 70/15/15 train/val/test masks (used when building the final model). 
    # It uses the train_test_split function from scikit-learn to split the indices of the labels into train, validation, and test sets, 
    # while preserving the class distribution of the labels in each set.
    idx = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(idx, test_size=0.3, stratify=labels, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=seed)
    return [_mask(len(labels), i) for i in (train_idx, val_idx, test_idx)]


def stratified_kfold_masks(labels, k=5, seed=42, groups=None):
    # returns a generator that yields k stratified train/val/test masks for cross-validation.
    # per run, the data is split into k folds, and for each fold, a train/val/test split is created while preserving the class distribution of the labels in each set.
    
    # random: if groups is 0, it uses StratifiedKFold to split the data into k folds, and then uses train_test_split to create a validation set from the training set.
    # group: if groups is not None, it uses StratifiedGroupKFold to split the data into k folds based on the group labels, 
    # and then uses train_test_split to create a validation set from the training set.
    
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
    # returns a dictionary with the number of test samples, number of scams, number of scams caught, recall, number of false alarms, 
    # and number of legitimate samples for both connected and isolated postings in the graph.
    # used to analyze the performance of the model on connected vs. isolated postings in the graph, 
    # which can provide insights into the effectiveness of the graph-based approach for fraud detection.
    
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