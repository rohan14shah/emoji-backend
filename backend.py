# backend.py
# -----------------------------
# Minimal FastAPI service for text->emoji.
# Loads emoji embedding artifacts (FAISS + metadata),
# embeds each sentence, retrieves top-K emojis, re-ranks by sentiment,
# and returns JSON your frontend can render.

import os
import re
from typing import List, Optional

import numpy as np
import pandas as pd
import faiss
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# -----------------------------
# Config (change if needed)
# -----------------------------
ART_DIR = os.environ.get("EMOJI_ART_DIR", "./models")  # where artifacts live
EMBEDDER_NAME = os.environ.get("EMBEDDER_NAME", "all-MiniLM-L6-v2")
TOP_K_DEFAULT = int(os.environ.get("TOP_K_DEFAULT", "5"))
MIN_SCORE_DEFAULT = float(os.environ.get("MIN_SCORE_DEFAULT", "0.20"))
SENTIMENT_WEIGHT_DEFAULT = float(os.environ.get("SENTIMENT_WEIGHT_DEFAULT", "0.30"))

# -----------------------------
# FastAPI app + CORS
# -----------------------------
app = FastAPI(title="Emoji Translator API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod (e.g., ["https://your-frontend.app"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Input schema
# -----------------------------
class AnnotateReq(BaseModel):
    text: str
    top_k: int = TOP_K_DEFAULT
    min_score: float = MIN_SCORE_DEFAULT
    sentiment_weight: float = SENTIMENT_WEIGHT_DEFAULT
    profile_id: Optional[str] = None  # reserved for personalization later

# -----------------------------
# Utilities
# -----------------------------
_sentence_splitter = re.compile(r'(?<=[.!?])\s+')

def split_sentences(paragraph: str) -> List[str]:
    paragraph = paragraph.strip()
    if not paragraph:
        return []
    # simple, fast sentence split; replace with spaCy if you prefer
    parts = [s.strip() for s in _sentence_splitter.split(paragraph) if s.strip()]
    return parts or [paragraph]

def normalize_l2(v: np.ndarray) -> np.ndarray:
    if v.ndim == 1:
        v = v.reshape(1, -1)
    denom = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return (v / denom).astype("float32")

# Placeholder sentiment (returns neutral). Swap with a real model if you want.
# Return in [-1, 1].
def text_sentiment_score(text: str) -> float:
    return 0.0

# -----------------------------
# Load artifacts on startup
# -----------------------------
@app.on_event("startup")
def _load_artifacts():
    global model, index, meta, emoji_vecs

    # model
    model = SentenceTransformer(EMBEDDER_NAME)

    # metadata
    meta_path = os.path.join(ART_DIR, "emoji_meta.parquet")
    meta = pd.read_parquet(meta_path)

    # rename columns if needed
    rename_map = {}
    if "Emoji" in meta.columns and "emoji" not in meta.columns:
        rename_map["Emoji"] = "emoji"
    if "Description" in meta.columns and "description" not in meta.columns:
        rename_map["Description"] = "description"
    if "Sentiment" in meta.columns and "sentiment" not in meta.columns:
        rename_map["Sentiment"] = "sentiment"
    if rename_map:
        meta.rename(columns=rename_map, inplace=True)

    # vectors
    vecs_path = os.path.join(ART_DIR, "emoji_vectors.npy")
    emoji_vecs = np.load(vecs_path).astype("float32")  # expected L2-normalized

    # FAISS index
    index_path = os.path.join(ART_DIR, "emoji.index")
    index = faiss.read_index(index_path)

    # quick assert
    if index.ntotal != emoji_vecs.shape[0]:
        raise RuntimeError("FAISS index size != number of vectors; rebuild artifacts.")
    if "emoji" not in meta.columns or "description" not in meta.columns:
        raise RuntimeError("Metadata missing required columns 'emoji' or 'description'.")

# -----------------------------
# Health
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True, "n_emojis": int(index.ntotal), "embedder": EMBEDDER_NAME}

# -----------------------------
# Main endpoint
# -----------------------------
@app.post("/annotate")
def annotate(req: AnnotateReq):
    sentences = split_sentences(req.text)
    if not sentences:
        return {"overall_sentiment": 0.0, "sentences": []}

    # Embed sentences (L2-normalized so cosine==inner product)
    X = model.encode(sentences, normalize_embeddings=True).astype("float32")

    results = []
    sentiments = []
    for i, sent in enumerate(sentences):
        x = X[i].reshape(1, -1)
        scores, ids = index.search(x, req.top_k)  # cosine via IP (vectors are normalized)
        scores, ids = scores[0], ids[0]

        s_score = float(text_sentiment_score(sent))
        sentiments.append(s_score)

        em = meta.iloc[ids].copy().reset_index(drop=True)
        em["semantic_score"] = scores

        # optional sentiment re-rank if column is present
        if "sentiment" in em.columns and em["sentiment"].notna().all():
            adj = 1 - req.sentiment_weight * np.abs(s_score - em["sentiment"].astype(float).values)
            em["final_score"] = em["semantic_score"].astype(float) * adj
        else:
            em["final_score"] = em["semantic_score"].astype(float)

        # filter by threshold
        em = em[em["final_score"] >= req.min_score].sort_values("final_score", ascending=False)

        results.append({
            "text": sent,
            "sentiment": s_score,
            "emojis": [
                {
                    "emoji": row.get("emoji", ""),
                    "description": row.get("description", ""),
                    "score": float(row["final_score"])
                }
                for _, row in em.iterrows()
            ]
        })

    overall = float(np.mean(sentiments)) if sentiments else 0.0
    return {"overall_sentiment": overall, "sentences": results}

# ----------------------------

# Run locally:
#   uvicorn backend:app --reload
# -----------------------------
