# build_model.py
from logger import RunLogger

import argparse
import hashlib
import json
import os
import time
from datetime import datetime

import torch
import torch_geometric
import transformers

from evelynn import FORMAT_VERSION, Evelynn
from models.graphsage_model import FraudDetectionGraphSAGE
from train_cv import CV_CONFIG, compute_pos_weight, load_inputs, train_one_model
from utils import (
    GNN_DEVICE, HARDWARE, K_NEIGHBORS, MAX_CLIQUE, TABULAR_COLS, TITLE_CAP,
    company_hash, iid_masks, set_seed,
)

ARCH = {"hidden_channels": 256, "out_channels": 128, "attn_dim": 128, "dropout": 0.4}
ENCODER = {"name": "distilroberta-base", "max_len": 512, "pooling": "mean"}

EXAMPLE_POSTINGS = [
    {   # should look legitimate
        "title": "Senior Software Engineer",
        "company_profile": "Northwind Systems builds logistics software used by freight companies in 40 countries.",
        "description": "Design and maintain backend services for our routing platform. Work with product and "
                       "operations teams, review code, and mentor junior engineers.",
        "requirements": "5+ years of experience with Python or Java, SQL, and cloud infrastructure.",
        "benefits": "Health insurance, paid leave, learning budget.",
        "telecommuting": 0, "has_company_logo": 1, "has_questions": 1, "salary_range": "90000-120000",
    },
    {   # classic advance-fee pattern
        "title": "Data Entry Clerk - Work From Home",
        "company_profile": "",
        "description": "Earn $5000 weekly typing from home, no experience needed. To start, pay a one-time "
                       "$99 registration fee and send your bank details to hiring.team.jobs@gmail.com",
        "requirements": "", "benefits": "",
        "telecommuting": 1, "has_company_logo": 0, "has_questions": 0, "salary_range": "",
    },
]


def build_reference_index(df):
    """Lookup tables used at inference to link a new posting to the reference graph."""
    company, title, contact = {}, {}, {}
    for i, (profile, norm_title, emails, phones) in enumerate(
        zip(df["company_profile"], df["normalized_title"], df["emails"], df["phones"])
    ):
        if profile != "":
            company.setdefault(company_hash(profile), []).append(i)
        if norm_title != "":
            title.setdefault(norm_title, []).append(i)
        for c in emails + phones:
            contact.setdefault(c, []).append(i)
    contact = {c: sorted(set(ids)) for c, ids in contact.items()}
    return {"company": company, "title": title, "contact": contact}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(out_path, seed, all_data):
    with RunLogger("build_model") as log:
        set_seed(seed)
        df, x_text, x_tab, y, edge_index = load_inputs(log)
        labels = df["label"].values

        train_m, val_m, test_m = (m.to(GNN_DEVICE) for m in iid_masks(labels, seed))
        if all_data:
            train_m, eval_m, eval_name = train_m | test_m, val_m, "validation (NOT held out)"
        else:
            eval_m, eval_name = test_m, "test (held out)"

        cfg = {**CV_CONFIG, "seed": seed}
        pos_weight = compute_pos_weight(y, train_m, cfg["pos_weight_cap"]).to(GNN_DEVICE)
        arch = {"text_dim": x_text.shape[1], "tabular_dim": x_tab.shape[1], "fusion_mode": "cross_attention", **ARCH}
        model = FraudDetectionGraphSAGE(**arch)

        t0 = time.time()
        metrics, threshold, _ = train_one_model(
            model, lambda m: m(x_text, edge_index, x_tab), y, train_m, val_m, eval_m,
            pos_weight, cfg, GNN_DEVICE,
        )
        log.event("train", "final model trained", seconds=round(time.time() - t0, 1),
                  threshold=threshold, evaluated_on=eval_name, **metrics)

        model.eval()
        with torch.no_grad():
            ref_logits = model(x_text, edge_index, x_tab).squeeze(1).cpu()

        n_nodes = int(x_text.shape[0])
        deg = torch.bincount(edge_index[0], minlength=n_nodes)
        payload = {
            "format_version": FORMAT_VERSION,
            "name": "evelynn",
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "architecture": arch,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "threshold": float(threshold),
            "encoder": ENCODER,
            "graph": {"max_clique": MAX_CLIQUE, "k_neighbors": K_NEIGHBORS, "title_cap": TITLE_CAP},
            "tabular_columns": TABULAR_COLS + ["log1p_degree"],
            "training": {"mode": "all_data" if all_data else "iid_70_15_15", "seed": seed, "config": cfg},
            "metrics": {"evaluated_on": eval_name, **{k: float(v) for k, v in metrics.items()}},
            "reference": {
                "x_text": x_text.cpu(), "x_tab": x_tab.cpu(), "edge_index": edge_index.cpu(),
                "y": y.cpu().long(), "index": build_reference_index(df),
            },
            "ref_logits": ref_logits,
        }
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        torch.save(payload, out_path)
        size_mb = round(os.path.getsize(out_path) / 1e6, 1)
        log.event("export", "saved", path=out_path, size_mb=size_mb)

        # --- verify the exported file the way a future user would load it ---
        model_loaded = Evelynn.load(out_path, device=GNN_DEVICE)
        self_test = model_loaded.self_test()
        log.event("verify", "reference-graph self test", **self_test)
        if not self_test["passed"]:
            raise RuntimeError(f"Exported model does not reproduce its own logits: {self_test}")

        examples = []
        for posting in EXAMPLE_POSTINGS:
            result = model_loaded.predict(posting)
            examples.append({"title": posting["title"], **result})
        log.event("verify", "example predictions", predictions=examples)

        card = {
            "name": "evelynn",
            "file": out_path,
            "sha256": sha256_of(out_path),
            "size_mb": size_mb,
            "built_at": payload["built_at"],
            "training_mode": payload["training"]["mode"],
            "decision_threshold": float(threshold),
            "metrics": payload["metrics"],
            "architecture": arch,
            "encoder": ENCODER,
            "reference_graph": {
                "nodes": n_nodes,
                "directed_edges": int(edge_index.shape[1]),
                "fraud_postings": int(y.sum().item()),
                "isolated_postings": int((deg == 0).sum().item()),
            },
            "training_config": cfg,
            "versions": {"torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
                         "transformers": transformers.__version__},
            "hardware": HARDWARE,
            "self_test": self_test,
            "example_predictions": examples,
            "usage": "from evelynn import Evelynn; Evelynn.load('outputs/evelynn.pt').predict(posting_dict)",
        }
        card_path = os.path.join(os.path.dirname(out_path) or ".", "evelynn_card.json")
        with open(card_path, "w") as f:
            json.dump(card, f, indent=2)
        log.finish("ok", model_file=out_path, model_card=card_path, sha256=card["sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the final Evelynn model (evelynn.pt).")
    parser.add_argument("--out", default=os.path.join("outputs", "evelynn.pt"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-data", action="store_true",
                        help="train on train+test (85%%); no held-out metrics are reported")
    args = parser.parse_args()
    build(args.out, args.seed, args.all_data)