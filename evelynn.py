# evelynn.py
import json
import math
import sys

import numpy as np
import torch

from models.distilroberta_encoder import DistilRobertaModule
from models.graphsage_model import FraudDetectionGraphSAGE
from utils import company_hash, posting_to_features

FORMAT_VERSION = 1


class Evelynn:
    def __init__(self, ckpt, device=None):
        if ckpt.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported evelynn.pt format: {ckpt.get('format_version')!r}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.threshold = float(ckpt["threshold"])
        self.meta = {k: ckpt[k] for k in ("name", "built_at", "encoder", "graph", "tabular_columns",
                                          "architecture", "training", "metrics")}

        arch = ckpt["architecture"]
        self.model = FraudDetectionGraphSAGE(
            text_dim=arch["text_dim"], tabular_dim=arch["tabular_dim"],
            hidden_channels=arch["hidden_channels"], out_channels=arch["out_channels"],
            attn_dim=arch["attn_dim"], dropout=arch["dropout"], fusion_mode=arch["fusion_mode"],
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

        ref = ckpt["reference"]
        self.ref_x = ref["x_text"].to(self.device)
        self.ref_tab = ref["x_tab"].to(self.device)
        self.ref_edges = ref["edge_index"].to(self.device)
        self.ref_y = ref["y"].to(self.device)
        self.index = ref["index"]
        self._saved_ref_logits = ckpt["ref_logits"]
        self._encoder = None

    @classmethod
    def load(cls, path="outputs/evelynn.pt", device=None):
        # weights_only=True: the file contains only tensors and plain Python containers, no pickled code
        return cls(torch.load(path, map_location="cpu", weights_only=True), device)

    # ------------------------------------------------------------------ scoring
    @torch.no_grad()
    def predict(self, posting):
        f = posting_to_features(posting)
        found = self._find_neighbors(f, np.random.default_rng(0))
        neighbor_ids = sorted(set(found["company"]) | set(found["title"]) | set(found["contact"]))

        n = self.ref_x.shape[0]
        emb = self._get_encoder().embed_texts([f["full_text"]], max_len=self.meta["encoder"]["max_len"])
        x_new = torch.tensor(emb, dtype=torch.float, device=self.device)
        tab_new = torch.tensor([f["tab"] + [math.log1p(len(neighbor_ids))]],
                               dtype=torch.float, device=self.device)

        ids = torch.tensor(neighbor_ids, dtype=torch.long, device=self.device)
        new = torch.full_like(ids, n)
        new_edges = torch.stack([torch.cat([new, ids]), torch.cat([ids, new])])

        logits = self.model(torch.cat([self.ref_x, x_new]),
                            torch.cat([self.ref_edges, new_edges], dim=1),
                            torch.cat([self.ref_tab, tab_new]))
        prob = torch.sigmoid(logits[n]).item()
        return {
            "probability": prob,
            "is_fraud": prob >= self.threshold,
            "threshold": self.threshold,
            "n_neighbors": len(neighbor_ids),
            "neighbors_by_link": {k: len(set(v)) for k, v in found.items()},
            "neighbors_known_fraud": int(self.ref_y[ids].sum().item()) if len(neighbor_ids) else 0,
        }

    def predict_batch(self, postings):
        return [self.predict(p) for p in postings]

    # ------------------------------------------------------------------ integrity check
    @torch.no_grad()
    def reference_logits(self):
        return self.model(self.ref_x, self.ref_edges, self.ref_tab).squeeze(1).cpu()

    def self_test(self, atol=1e-3):
        """Re-scores the stored reference graph and compares with the logits saved at build time."""
        diff = (self.reference_logits() - self._saved_ref_logits).abs().max().item()
        return {"max_abs_logit_diff": diff, "passed": diff <= atol}

    # ------------------------------------------------------------------ internals
    def _get_encoder(self):
        if self._encoder is None:
            self._encoder = DistilRobertaModule(self.device, self.meta["encoder"]["name"])
        return self._encoder

    def _cap(self, ids, rng):
        """Same rule as training: in a group of more than MAX_CLIQUE postings a node links to K random members."""
        g = self.meta["graph"]
        if len(ids) + 1 > g["max_clique"]:
            return rng.choice(ids, size=g["k_neighbors"], replace=False).tolist()
        return list(ids)

    def _find_neighbors(self, f, rng):
        g = self.meta["graph"]
        found = {"company": [], "title": [], "contact": []}

        if f["company_profile"] != "":
            ids = self.index["company"].get(company_hash(f["company_profile"]), [])
            found["company"] = self._cap(ids, rng) if ids else []

        ids = self.index["title"].get(f["norm_title"], []) if f["norm_title"] else []
        if 1 <= len(ids) <= g["title_cap"] - 1:   # the title (including this posting) is used <= title_cap times
            found["title"] = self._cap(ids, rng)

        for contact in sorted(set(f["contacts"])):
            ids = self.index["contact"].get(contact, [])
            if ids:
                found["contact"] += self._cap(ids, rng)
        return found


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python evelynn.py posting.json [outputs/evelynn.pt]")
    with open(sys.argv[1], encoding="utf-8") as fh:
        posting = json.load(fh)
    model = Evelynn.load(sys.argv[2] if len(sys.argv) > 2 else "outputs/evelynn.pt")
    print(json.dumps(model.predict(posting), indent=2))