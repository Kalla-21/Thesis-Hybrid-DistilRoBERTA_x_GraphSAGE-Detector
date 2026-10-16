import hashlib
import json
import os
import re
import time

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torch_geometric.utils import k_hop_subgraph

# Import your model architectures
from model.graphsage_model import FraudDetectionGraphSAGE
from model.distilroberta_encoder import DistilRobertaModule

app = FastAPI(title="Hybrid Scam Detection API")

# Allow the frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HERE = os.path.dirname(os.path.abspath(__file__))

# --- MODEL SETUP ---
TEXT_DIM = 768
DEVICE = "cpu"

print("Loading models...")
HERE = os.path.dirname(os.path.abspath(__file__))

_assets = np.load(os.path.join(HERE, "model", "graph_assets.npz"))
N = _assets["embeddings"].shape[0]
X_TEXT = torch.from_numpy(_assets["embeddings"].astype(np.float32))
X_TAB = torch.from_numpy(_assets["tabular"])
EDGE_INDEX = torch.from_numpy(_assets["edge_index"].astype(np.int64))
COMPANY_HASH = _assets["company_hash"]
TITLE_HASH = _assets["title_hash"]

TABULAR_DIM = X_TAB.shape[1]  # Read dynamic dimension from graph assets (11)

scam_model = FraudDetectionGraphSAGE(text_dim=TEXT_DIM, tabular_dim=TABULAR_DIM, fusion_mode="cross_attention")
scam_model.load_state_dict(torch.load(os.path.join(HERE, "model", "best_cross_attention_model.pt"),
                                      map_location=torch.device('cpu'), weights_only=True))
scam_model.eval()

# Load the DistilRoBERTa translator
text_encoder = DistilRobertaModule(DEVICE)

# --- INFERENCE CONFIG ---
with open(os.path.join(HERE, "model", "inference_config.json")) as f:
    CONFIG = json.load(f)
THRESHOLD = CONFIG["threshold"]
MAX_LENGTH = CONFIG["max_length"]
MAX_TITLE_GROUP = CONFIG["max_title_group"]

# --- TRAINING GRAPH ---
# The model was trained transductively on one big graph of the dataset's postings (edges = same
# company profile, or same title when it appears 2-15 times). A new posting is added to that graph
# as one extra node, linked the same way, so the GraphSAGE branch sees real neighbours.
# Exported by build_graph_assets.py.

_assets = np.load(os.path.join(HERE, "model", "graph_assets.npz"))
N = _assets["embeddings"].shape[0]
X_TEXT = torch.from_numpy(_assets["embeddings"].astype(np.float32))
X_TAB = torch.from_numpy(_assets["tabular"])
EDGE_INDEX = torch.from_numpy(_assets["edge_index"].astype(np.int64))
COMPANY_HASH = _assets["company_hash"]
TITLE_HASH = _assets["title_hash"]
# optional, added by add_neighbour_info.py: lets the neighbour picture show titles and scam/legit labels
LABELS = _assets["labels"] if "labels" in _assets.files else None
TITLES = _assets["titles"] if "titles" in _assets.files else None
_title_values, _title_counts = np.unique(TITLE_HASH, return_counts=True)
TITLE_COUNT = dict(zip(_title_values.tolist(), _title_counts.tolist()))
print(f"Models loaded successfully! Graph: {N} nodes, {EDGE_INDEX.shape[1]} edges")

FIELDS = ["title", "company_profile", "description", "requirements", "benefits"]
FLAGS = ["telecommuting", "has_company_logo", "has_questions"]
MAX_SENTENCES = 40      # most sentences re-scored in one explanation
MAX_SHOWN_NEIGHBOURS = 30


def h64(s: str) -> int:
    """Same hash as build_graph_assets.py (0 = empty string)."""
    if s == "":
        return 0
    return int(np.frombuffer(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), dtype=np.uint64)[0])


def find_neighbours(title: str, company_profile: str):
    """Dataset nodes the new posting would be linked to, using the training edge rules.

    Returns (all, by_company, by_title) as index arrays.
    """
    by_company = np.zeros(N, dtype=bool)
    by_title = np.zeros(N, dtype=bool)

    company = h64(company_profile)
    if company != 0:
        by_company = COMPANY_HASH == np.uint64(company)

    title_h = h64(title)
    existing = TITLE_COUNT.get(title_h, 0)
    # training rule: title groups of size 2..15 are connected; the new node makes it existing + 1
    if title_h != 0 and 1 <= existing <= MAX_TITLE_GROUP - 1:
        by_title = TITLE_HASH == np.uint64(title_h)

    return np.flatnonzero(by_company | by_title), np.flatnonzero(by_company), np.flatnonzero(by_title)


class Subgraph:
    """The new posting plus everything within 3 hops of it.

    The model has 3 GraphSAGE layers, so the new node's output depends only on nodes within 3 hops;
    scoring that neighbourhood gives the same answer as scoring the whole graph, much faster. The text
    embedding and flags of the new node are passed in per call, so variants of one posting (fields
    removed, flags flipped, ...) can be re-scored against the same structure.
    """

    def __init__(self, neighbours: np.ndarray):
        if len(neighbours):
            nb = torch.from_numpy(neighbours).long()
            new = torch.full_like(nb, N)
            edge_index = torch.cat([EDGE_INDEX, torch.stack([nb, new]), torch.stack([new, nb])], dim=1)
            subset, self.sub_edge_index, _, _ = k_hop_subgraph(
                N, 3, edge_index, relabel_nodes=True, num_nodes=N + 1)
            assert int(subset[-1]) == N  # sorted, and the new node has the largest id
            old = subset[:-1]
        else:
            old = torch.zeros(0, dtype=torch.long)
            self.sub_edge_index = torch.zeros((2, 0), dtype=torch.long)
        self.x_old = X_TEXT[old]
        self.tab_old = X_TAB[old]
        self.center = len(old)
        self.size = len(old) + 1

    def score(self, embedding: torch.Tensor, tab: torch.Tensor) -> float:
        x = torch.cat([self.x_old, embedding.reshape(1, -1)])
        t = torch.cat([self.tab_old, tab.reshape(1, -1)])
        with torch.no_grad():
            return torch.sigmoid(scam_model(x, self.sub_edge_index, t)[self.center]).item()

    def score_without_graph(self, embedding: torch.Tensor, tab: torch.Tensor) -> float:
        """Same posting, but linked to nothing."""
        with torch.no_grad():
            logit = scam_model(embedding.reshape(1, -1), torch.zeros((2, 0), dtype=torch.long), tab.reshape(1, -1))
        return torch.sigmoid(logit[0]).item()


def build_text(fields: dict) -> str:
    # identical to the concatenation used for training
    return " ".join(fields[f] for f in FIELDS)


def tab_tensor(job) -> torch.Tensor:
    # 3 core user inputs from UI
    flags = [float(job.telecommuting), float(job.has_company_logo), float(job.has_questions)]
    
    # Pad remaining tabular features with 0.0 if TABULAR_DIM > 3
    if len(flags) < TABULAR_DIM:
        flags += [0.0] * (TABULAR_DIM - len(flags))
        
    return torch.tensor(flags[:TABULAR_DIM], dtype=torch.float)

# --- DATA SCHEMA (User Experience) ---
class JobInput(BaseModel):
    title: str
    company_profile: str
    description: str
    requirements: str
    benefits: str
    telecommuting: int
    has_company_logo: int
    has_questions: int

# --- PREDICTION ENDPOINT ---
@app.post("/predict")
def predict_scam(job: JobInput):
    try:
        # 1. Convert Text to 768D Embedding (same concatenation and max_length as training)
        fields = {f: getattr(job, f) for f in FIELDS}
        embedding = text_encoder.extract_single_embedding(build_text(fields), max_length=MAX_LENGTH)
        embedding = torch.tensor(embedding, dtype=torch.float).squeeze(0)

        # 2. Link the new node into the training graph (undirected; no self-loop, as in training)
        neighbours, _, _ = find_neighbours(job.title, job.company_profile)

        # 3. Predict
        score = Subgraph(neighbours).score(embedding, tab_tensor(job))

        return {
            "status": "success",
            "score": round(score, 4),
            "threshold": round(THRESHOLD, 4),
            "is_scam": score >= THRESHOLD,
            "linked_postings": int(len(neighbours)),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- EXPLANATION ENDPOINT ---
# Every number below is "score with the posting as written" minus "score with one thing changed".
# Positive = that part pushes the posting toward "scam". It is a counterfactual explanation of what
# moves the score, not a look inside the network.
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def sentence_spans(text: str, offset: int):
    """(start, end) character spans of the sentences / bullet lines in one field, in combined-text coordinates."""
    spans, start = [], 0
    for m in list(_SPLIT.finditer(text)) + [None]:
        end = m.start() if m else len(text)
        chunk = text[start:end]
        lead = len(chunk) - len(chunk.lstrip())
        chunk = chunk.strip()
        if chunk:
            spans.append((offset + start + lead, offset + start + lead + len(chunk)))
        if m:
            start = m.end()
    return spans


def read_cutoff(text: str):
    """Character position where the encoder stops reading (it only keeps the first MAX_LENGTH tokens)."""
    enc = text_encoder.tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_offsets_mapping=True)
    ends = [e for s, e in enc["offset_mapping"] if e > s]
    cutoff = max(ends) if ends else 0
    return cutoff, cutoff < len(text.rstrip())


@app.post("/explain")
def explain_prediction(job: JobInput):
    try:
        t0 = time.time()
        fields = {f: getattr(job, f) for f in FIELDS}
        text = build_text(fields)
        tab = tab_tensor(job)

        all_nb, company_nb, title_nb = find_neighbours(job.title, job.company_profile)
        graph = Subgraph(all_nb)
        cutoff, truncated = read_cutoff(text)

        # where each field sits in the combined text
        field_span, pos = {}, 0
        for f in FIELDS:
            field_span[f] = (pos, pos + len(fields[f]))
            pos += len(fields[f]) + 1

        # sentences the model can actually see (the rest is cut off before encoding)
        sentences = []
        for f in FIELDS:
            for s, e in sentence_spans(fields[f], field_span[f][0]):
                sentences.append({"field": f, "start": s, "end": e, "text": text[s:e], "read": s < cutoff})
        scored = [s for s in sentences if s["read"]][:MAX_SENTENCES]

        # one batch of variant texts: base, each non-empty field removed, each readable sentence removed
        variants = [text]
        present = [f for f in FIELDS if fields[f].strip()]
        variants += [build_text({**fields, f: ""}) for f in present]
        variants += [text[:s["start"]] + text[s["end"]:] for s in scored]
        embeddings = text_encoder.extract_batch_embeddings(variants, max_length=MAX_LENGTH)

        base = graph.score(embeddings[0], tab)

        # 1. fields
        field_rows = []
        for i, f in enumerate(present):
            without = graph.score(embeddings[1 + i], tab)
            s, e = field_span[f]
            read = max(0, min(e, cutoff) - s) / max(1, e - s)
            field_rows.append({"field": f, "chars": e - s, "read_fraction": round(read, 3),
                               "score_without": round(without, 4), "delta": round(base - without, 4)})
        for f in FIELDS:
            if f not in present:
                field_rows.append({"field": f, "chars": 0, "empty": True})
        field_rows.sort(key=lambda r: FIELDS.index(r["field"]))

        # 2. sentences
        for i, sent in enumerate(scored):
            without = graph.score(embeddings[1 + len(present) + i], tab)
            sent["delta"] = round(base - without, 4)

        # 3. flags
        flag_rows = []
        for i, name in enumerate(FLAGS):
            flipped = tab.clone()
            flipped[i] = 1.0 - flipped[i]
            without = graph.score(embeddings[0], flipped)
            flag_rows.append({"flag": name, "value": int(tab[i]), "score_if_flipped": round(without, 4),
                              "delta": round(base - without, 4)})

        # 4. graph
        no_graph = graph.score_without_graph(embeddings[0], tab)
        company_set, title_set = set(company_nb.tolist()), set(title_nb.tolist())
        shown = all_nb
        if len(shown) > MAX_SHOWN_NEIGHBOURS:
            shown = np.sort(np.random.default_rng(0).choice(shown, MAX_SHOWN_NEIGHBOURS, replace=False))
        graph_info = {
            "linked": int(len(all_nb)),
            "by_company": int(len(company_nb)),
            "by_title": int(len(title_nb)),
            "subgraph_nodes": graph.size,
            "score_with_graph": round(base, 4),
            "score_without_graph": round(no_graph, 4),
            "delta": round(base - no_graph, 4),
            "has_labels": LABELS is not None,
            "scam_neighbours": int(LABELS[all_nb].sum()) if LABELS is not None else None,
            "shown": [{
                "title": str(TITLES[i]) if TITLES is not None else None,
                "label": int(LABELS[i]) if LABELS is not None else None,
                "reasons": [r for r, s in (("same company", company_set), ("same title", title_set)) if int(i) in s],
            } for i in shown],
        }

        return {
            "status": "success",
            "score": round(base, 4),
            "threshold": round(THRESHOLD, 4),
            "is_scam": base >= THRESHOLD,
            "read_limit": {"max_tokens": MAX_LENGTH, "cutoff_char": int(cutoff), "truncated": bool(truncated)},
            "fields": field_rows,
            "sentences": sentences,
            "flags": flag_rows,
            "graph": graph_info,
            "seconds": round(time.time() - t0, 2),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
