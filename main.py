import os
import re
import time
import traceback
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from evelynn import Evelynn

app = FastAPI(title="Hybrid Scam Detection API (Evelynn)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model", "evelynn.pt")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(HERE, "outputs", "evelynn.pt")

print(f"Loading Evelynn engine from {MODEL_PATH}...")
evelynn_engine = Evelynn.load(MODEL_PATH)

test_res = evelynn_engine.self_test()
print(f"Evelynn loaded successfully! Reference logit check passed: {test_res['passed']} (max diff: {test_res['max_abs_logit_diff']:.6f})")


class JobInput(BaseModel):
    title: str
    company_profile: Optional[str] = ""
    description: Optional[str] = ""
    requirements: Optional[str] = ""
    benefits: Optional[str] = ""
    telecommuting: int = 0
    has_company_logo: int = 0
    has_questions: int = 0


FIELDS = ["title", "company_profile", "description", "requirements", "benefits"]
FLAGS = ["telecommuting", "has_company_logo", "has_questions"]
# fields whose sentences are scored one by one (the title is scored as a whole field)
SENTENCE_FIELDS = ["description", "requirements", "company_profile", "benefits"]
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _pieces(text):
    """Split text into (start, end, is_sentence) spans that cover it exactly, in order.
    A sentence is a piece between splits whose stripped length is over 5 characters;
    whitespace, separators and shorter fragments are returned as unscored spans."""
    spans, pos = [], 0
    bounds = [(m.start(), m.end()) for m in _SPLIT.finditer(text)] + [(len(text), len(text))]
    for sep_start, sep_end in bounds:
        piece = text[pos:sep_start]
        s = pos + (len(piece) - len(piece.lstrip()))
        e = sep_start - (len(piece) - len(piece.rstrip()))
        if e - s > 5:
            if s > pos:
                spans.append((pos, s, False))
            spans.append((s, e, True))
            if sep_start > e:
                spans.append((e, sep_start, False))
        elif sep_start > pos:
            spans.append((pos, sep_start, False))
        if sep_end > sep_start:
            spans.append((sep_start, sep_end, False))
        pos = sep_end
    return spans


@app.post("/predict")
def predict_scam(job: JobInput):
    try:
        posting = job.dict()
        res = evelynn_engine.predict(posting)
        prob = round(res["probability"], 4)
        thresh = round(res["threshold"], 4)
        is_fraud = bool(res["is_fraud"])
        verdict_str = "Fraudulent Job Posting" if is_fraud else "Legitimate Job Posting"

        return {
            "status": "success",
            "score": prob,
            "probability": prob,
            "scam_probability": prob,
            "threshold": thresh,
            "is_scam": is_fraud,
            "is_fraud": is_fraud,
            "verdict": verdict_str,
            "linked_postings": res["n_neighbors"],
            "neighbors_by_link": res["neighbors_by_link"],
            "neighbors_known_fraud": res["neighbors_known_fraud"],
        }
    except Exception as e:
        print(f"Error in /predict: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/explain")
def explain_prediction(job: JobInput):
    try:
        t0 = time.time()
        posting = job.dict()
        
        # Base prediction
        base_res = evelynn_engine.predict(posting)
        base_score = round(base_res["probability"], 4)
        thresh = round(base_res["threshold"], 4)
        is_fraud = bool(base_res["is_fraud"])
        verdict_str = "Fraudulent Job Posting" if is_fraud else "Legitimate Job Posting"

        fields = {f: getattr(job, f, "") or "" for f in FIELDS}

        # 1. Field-level ablation
        field_rows = []
        fields_dict = {}
        for f in FIELDS:
            val = fields[f].strip()
            if not val:
                field_rows.append({"field": f, "chars": 0, "empty": True, "delta": 0.0, "score_without": base_score})
                fields_dict[f] = 0.0
                continue
            
            mod_posting = posting.copy()
            mod_posting[f] = ""
            score_without = round(evelynn_engine.predict(mod_posting)["probability"], 4)
            delta = round(base_score - score_without, 4)
            
            field_rows.append({
                "field": f,
                "chars": len(val),
                "score_without": score_without,
                "delta": delta
            })
            fields_dict[f] = delta

        # 2. Flag flips
        flag_rows = []
        flags_dict = {}
        for flag in FLAGS:
            mod_posting = posting.copy()
            mod_posting[flag] = 1 - int(mod_posting.get(flag, 0))
            score_if_flipped = round(evelynn_engine.predict(mod_posting)["probability"], 4)
            delta = round(base_score - score_if_flipped, 4)
            
            flag_rows.append({
                "flag": flag,
                "value": int(posting.get(flag, 0)),
                "score_if_flipped": score_if_flipped,
                "delta": delta
            })
            flags_dict[flag] = delta

        # 3. Sentence-level ablation: every sentence is removed in turn and the posting re-scored.
        # Each field also comes back as "segments" that rebuild its text exactly, so the page
        # can colour the posting in place; gaps and short fragments carry no score.
        sentence_rows = []
        segments = {}
        for field in SENTENCE_FIELDS:
            text = posting.get(field) or ""
            segs = []
            for s, e, is_sentence in _pieces(text):
                chunk = text[s:e]
                if not is_sentence:
                    if segs and "delta" not in segs[-1]:
                        segs[-1]["text"] += chunk
                    else:
                        segs.append({"text": chunk})
                    continue

                mod_posting = posting.copy()
                mod_posting[field] = text[:s] + text[e:]
                score_without = round(evelynn_engine.predict(mod_posting)["probability"], 4)
                delta = round(base_score - score_without, 4)

                segs.append({"text": chunk, "score_without": score_without, "delta": delta})
                sentence_rows.append({
                    "field": field,
                    "sentence": chunk,
                    "text": chunk,
                    "content": chunk,
                    "score_without": score_without,
                    "delta": delta,
                    "impact": delta,
                    "truncated": False
                })
            segments[field] = segs

        # the title is graded as one piece, with its field-level value
        title_row = next(r for r in field_rows if r["field"] == "title")
        title_text = posting.get("title") or ""
        segments["title"] = ([{"text": title_text, "score_without": title_row["score_without"], "delta": title_row["delta"]}]
                             if not title_row.get("empty") else [{"text": title_text}])

        is_truncated = False
        sentences_payload = {
            "truncated": is_truncated,
            "count": len(sentence_rows),
            "total": len(sentence_rows),
            "items": sentence_rows,
            "list": sentence_rows,
            "sentences": sentence_rows
        }

        # Narrative text summary
        summary = f"Classification Verdict: {verdict_str}\n"
        summary += f"Fraud Risk Score: {base_score:.2%} (Cutoff: {thresh:.2%})\n\n"
        summary += f"Graph Context:\n"
        summary += f" • Linked Neighbors: {base_res['n_neighbors']}\n"
        summary += f" • Fraudulent Neighbors: {base_res['neighbors_known_fraud']}\n\n"
        summary += "Field Importance (Score Change on Ablation):\n"
        for row in field_rows:
            if not row.get("empty"):
                summary += f" • {row['field'].replace('_', ' ').title()}: {row['delta']:+.2%}\n"

        explanation_list = [line for line in summary.split("\n") if line.strip()]

        return {
            "status": "success",
            "score": base_score,
            "probability": base_score,
            "scam_probability": base_score,
            "threshold": thresh,
            "is_scam": is_fraud,
            "is_fraud": is_fraud,
            "verdict": verdict_str,
            "truncated": is_truncated,
            
            # Sentence payloads
            "sentences": sentences_payload,
            "sentence_list": sentence_rows,
            "sentence_items": sentence_rows,
            "segments": segments,

            # Text / Summary variations
            "explanation": summary,
            "explanation_text": summary,
            "explanations": explanation_list,
            "text": summary,
            "summary": summary,
            "details": summary,
            
            # Field representations (List + Dict)
            "fields": field_rows,
            "field_impacts": fields_dict,
            "field_ablation": fields_dict,
            "ablation": fields_dict,
            "feature_importance": fields_dict,
            
            # Flag representations (List + Dict)
            "flags": flag_rows,
            "flag_flips": flags_dict,
            
            # Graph representations
            "graph": {
                "linked": base_res["n_neighbors"],
                "n_neighbors": base_res["n_neighbors"],
                "linked_postings": base_res["n_neighbors"],
                "by_link": base_res["neighbors_by_link"],
                "neighbors_by_link": base_res["neighbors_by_link"],
                "scam_neighbours": base_res["neighbors_known_fraud"],
                "neighbors_known_fraud": base_res["neighbors_known_fraud"],
            },
            "linked_postings": base_res["n_neighbors"],
            "neighbors_by_link": base_res["neighbors_by_link"],
            "neighbors_known_fraud": base_res["neighbors_known_fraud"],
            
            "seconds": round(time.time() - t0, 2),
        }
    except Exception as e:
        print(f"Error in /explain: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))