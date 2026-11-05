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
    def __init__(self, ckpt, device=None): # The constructor initializes the Evelynn model by loading the checkpoint, 
        #setting the device, and preparing the model architecture, encoder, and reference graph for inference.
        if ckpt.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported evelynn.pt format: {ckpt.get('format_version')!r}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.threshold = float(ckpt["threshold"])
        self.meta = {k: ckpt[k] for k in ("name", "built_at", "encoder", "graph", "tabular_columns",
                                          "architecture", "training", "metrics")}

        # Load the model architecture and state dictionary from the checkpoint,
        # and move the model to the specified device for inference.
        arch = ckpt["architecture"]
        
        # Initialize the FraudDetectionGraphSAGE model with the specified architecture
        self.model = FraudDetectionGraphSAGE(
            text_dim=arch["text_dim"], tabular_dim=arch["tabular_dim"],
            hidden_channels=arch["hidden_channels"], out_channels=arch["out_channels"],
            attn_dim=arch["attn_dim"], dropout=arch["dropout"], fusion_mode=arch["fusion_mode"],
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

        # Load the reference graph data (text embeddings, tabular features, edge indices, labels, and index mapping)
        # from the checkpoint, and move them to the specified device for inference.
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
    @torch.no_grad() # disables gradient computation for the following methods, which reduces memory usage and speeds up inference.
    def predict(self, posting):
     # Given a job posting, this method extracts its features, finds its neighbors in the reference graph, 
     # computes the model's prediction (probability of fraud), and returns a dictionary containing the 
     # prediction results and relevant information about the neighbors.
        f = posting_to_features(posting)
        found = self._find_neighbors(f, np.random.default_rng(0))
        neighbor_ids = sorted(set(found["company"]) | set(found["title"]) | set(found["contact"]))

        # Compute the number of nodes in the reference graph and their degrees based on the edge indices,
        # which will be included in the model metadata for reference.
        n = self.ref_x.shape[0]
        emb = self._get_encoder().embed_texts([f["full_text"]], max_len=self.meta["encoder"]["max_len"])
        x_new = torch.tensor(emb, dtype=torch.float, device=self.device)
        tab_new = torch.tensor([f["tab"] + [math.log1p(len(neighbor_ids))]],
                               dtype=torch.float, device=self.device)

        # Create new edges connecting the new posting to its neighbors in the reference graph,
        # and compute the model's logits for the new posting based on the updated graph structure.
        ids = torch.tensor(neighbor_ids, dtype=torch.long, device=self.device)
        new = torch.full_like(ids, n)
        new_edges = torch.stack([torch.cat([new, ids]), torch.cat([ids, new])])

        # Compute the model's logits for the new posting based on the updated graph structure,
        # and apply the sigmoid function to obtain the probability of fraud.
        logits = self.model(torch.cat([self.ref_x, x_new]),
                            torch.cat([self.ref_edges, new_edges], dim=1),
                            torch.cat([self.ref_tab, tab_new]))
        
        # Compute the probability of fraud for the new posting by applying the sigmoid function to the model's logits,
        # and return a dictionary containing the prediction results and relevant information about the neighbors.
        prob = torch.sigmoid(logits[n]).item()
        return {
            "probability": prob,
            "is_fraud": prob >= self.threshold,
            "threshold": self.threshold,
            "n_neighbors": len(neighbor_ids),
            "neighbors_by_link": {k: len(set(v)) for k, v in found.items()},
            "neighbors_known_fraud": int(self.ref_y[ids].sum().item()) if len(neighbor_ids) else 0,
        }

    def predict_batch(self, postings): # Given a list of job postings, 
        #this method applies the predict method to each posting in the list and returns a list of prediction results for all postings.
        return [self.predict(p) for p in postings]

    # ------------------------------------------------------------------ integrity check
    @torch.no_grad() 
    def reference_logits(self): # Computes the model's logits for the stored reference graph using the current model parameters,
        return self.model(self.ref_x, self.ref_edges, self.ref_tab).squeeze(1).cpu()

    def self_test(self, atol=1e-3): # Rescores the reference graph using the current model parameters and compares the 
        #results to the stored reference logits, returning a dictionary indicating whether the maximum 
        # absolute difference is within the specified tolerance.
        diff = (self.reference_logits() - self._saved_ref_logits).abs().max().item()
        return {"max_abs_logit_diff": diff, "passed": diff <= atol}

    # ------------------------------------------------------------------ internals
    def _get_encoder(self): # Lazily initializes and returns the DistilRobertaModule encoder for text embeddings,
        # loading it from the checkpoint if it hasn't been created yet.
        if self._encoder is None:
            self._encoder = DistilRobertaModule(self.device, self.meta["encoder"]["name"])
        return self._encoder

    def _cap(self, ids, rng): # Given a list of neighbor IDs and a random number generator, 
             # this method caps the number of neighbors based on the model's graph configuration.
        # Same rule as training: in a group of more than MAX_CLIQUE postings a node links to K random members.
        g = self.meta["graph"]
        if len(ids) + 1 > g["max_clique"]:
            return rng.choice(ids, size=g["k_neighbors"], replace=False).tolist()
        return list(ids)

    def _find_neighbors(self, f, rng): # Given a job posting's features and a random number generator, 
        # this method finds the neighbors of the posting in the reference graph
        g = self.meta["graph"]
        found = {"company": [], "title": [], "contact": []}

        if f["company_profile"] != "": # If the job posting has a non-empty company profile, 
            #this method looks up the company hash in the reference graph's index and retrieves the corresponding neighbor IDs.
            ids = self.index["company"].get(company_hash(f["company_profile"]), [])
            found["company"] = self._cap(ids, rng) if ids else []

        ids = self.index["title"].get(f["norm_title"], []) if f["norm_title"] else [] # If the job posting has a non-empty normalized title,
        # this method looks up the normalized title in the reference graph's index and retrieves the corresponding neighbor IDs, 
        # capping the number of neighbors based on the model's graph configuration.
        if 1 <= len(ids) <= g["title_cap"] - 1:   # the title (including this posting) is used <= title_cap times
            found["title"] = self._cap(ids, rng)

        for contact in sorted(set(f["contacts"])): # If the job posting has one or more unique contact identifiers, 
            # this method looks up each contact in the reference graph's index and retrieves the corresponding neighbor IDs, 
            # capping the number of neighbors based on the model's graph configuration.
            ids = self.index["contact"].get(contact, [])
            if ids: # If there are any neighbor IDs found for the contact, 
                # this method adds them to the list of found neighbors for the "contact" key in the found dictionary, 
                # capping the number of neighbors based on the model's graph configuration.
                found["contact"] += self._cap(ids, rng)
        return found


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python evelynn.py posting.json [outputs/evelynn.pt]")
    with open(sys.argv[1], encoding="utf-8") as fh:
        posting = json.load(fh)
    model = Evelynn.load(sys.argv[2] if len(sys.argv) > 2 else "outputs/evelynn.pt")
    print(json.dumps(model.predict(posting), indent=2))