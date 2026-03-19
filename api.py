"""
CineMind Movie Recommendation API
FastAPI backend that loads model and serves
recommendations based on movieIDs selected by the user in the frontend.

Run:
    uvicorn api:app --reload --host 0.0.0.0 --port 5000
    # (Production) uvicorn api:app --host 0.0.0.0 --port $PORT
"""

import os
import logging
import threading
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import pandas as pd
import pyarrow.dataset as pa_ds
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cinemind")

# Config
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = "models/model_weights.pth"          # root of moviesys/
RATINGS_PATH  = "data/processed/ratings.parquet"    # Optional: only needed for /api/movies/popular endpoint
MOVIES_PATH   = "data/raw/movies.csv"               # moviesys/dataset/movies.csv
EMBEDDING_DIM = 64                                  # must match training (fallback if not in checkpoint)
TOP_K_DEFAULT = 10                                  # recommendations returned by default


# --------------------------------------------------------------------------- #
# Low-memory helpers
def load_user_item_mappings_from_ratings_parquet_low_memory(ratings_path: str):
    """
    Reconstruct userId/movieId -> embedding index mappings without loading the
    full parquet into a pandas DataFrame (memory-safe fallback for legacy
    checkpoints).
    """
    if not os.path.exists(ratings_path):
        raise RuntimeError(f"ratings.parquet not found at: {ratings_path}")

    log.info("Loading user/item mappings from %s (low-memory mode)…", ratings_path)
    dataset = pa_ds.dataset(ratings_path, format="parquet")

    users = set()
    items = set()

    # Stream batches and keep only unique IDs.
    for batch in dataset.to_batches(columns=["userId", "movieId"], batch_size=65536):
        user_idx = batch.schema.get_field_index("userId")
        item_idx = batch.schema.get_field_index("movieId")
        user_col = batch.column(user_idx).to_numpy(zero_copy_only=False)
        item_col = batch.column(item_idx).to_numpy(zero_copy_only=False)
        users.update(np.unique(user_col).tolist())
        items.update(np.unique(item_col).tolist())

    unique_users = sorted(users)
    unique_items = sorted(items)

    user_to_idx = {u: i for i, u in enumerate(unique_users)}
    item_to_idx = {m: i for i, m in enumerate(unique_items)}
    return user_to_idx, item_to_idx



# Model definition  
class MatrixFactorizationWithBias(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 50):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.user_bias  = nn.Embedding(num_users, 1)
        self.item_bias  = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user_ids, item_ids):
        user_vecs   = self.user_embeddings(user_ids)
        item_vecs   = self.item_embeddings(item_ids)
        dot_product = (user_vecs * item_vecs).sum(dim=1)
        user_b = self.user_bias(user_ids).squeeze()
        item_b = self.item_bias(item_ids).squeeze()
        return dot_product + user_b + item_b + self.global_bias



# Recommender 
class MovieRecommender:
    """Thin wrapper around the trained model for inference."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Device: %s", self.device)

        # Load movies for titles
        log.info("Loading movies from %s …", MOVIES_PATH)
        self.movies_df = pd.read_csv(MOVIES_PATH)

        # Load model checkpoint
        log.info("Loading model checkpoint from %s …", MODEL_PATH)
        checkpoint = torch.load(MODEL_PATH, map_location=self.device)
        
        # Check if checkpoint contains metadata (full checkpoint) or just state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Full checkpoint with metadata
            log.info("Loading mappings from checkpoint metadata…")
            self.item_to_idx = checkpoint['item_to_idx']
            self.user_to_idx = checkpoint.get('user_to_idx', {})
            self.num_users = checkpoint['num_users']
            self.num_items = checkpoint['num_items']
            embedding_dim = checkpoint.get('embedding_dim', EMBEDDING_DIM)
            state_dict = checkpoint['model_state_dict']
            self.ratings_df = None  # Not needed when using full checkpoint
        else:
            # Legacy format: just state_dict, need to reconstruct from ratings.parquet
            log.warning(
                "Checkpoint appears to be state_dict only. Reconstructing mappings from ratings.parquet (low-memory mode)…"
            )
            if not os.path.exists(RATINGS_PATH):
                raise RuntimeError(
                    f"Model checkpoint doesn't contain metadata and {RATINGS_PATH} is missing. "
                    "Please retrain the model using the save_model() method that includes metadata."
                )
            # Keep memory usage low: only load unique user/item IDs, not the full ratings table.
            self.ratings_df = None
            self.user_to_idx, self.item_to_idx = load_user_item_mappings_from_ratings_parquet_low_memory(RATINGS_PATH)
            self.num_users = len(self.user_to_idx)
            self.num_items = len(self.item_to_idx)

            # Try to infer embedding_dim from the state_dict (avoids mismatch).
            embedding_dim = EMBEDDING_DIM
            for key in ("user_embeddings.weight", "module.user_embeddings.weight", "model.user_embeddings.weight"):
                t = checkpoint.get(key)
                if t is not None and getattr(t, "ndim", 0) == 2:
                    embedding_dim = int(t.shape[1])
                    break
            state_dict = checkpoint

        # Build reverse mapping
        self.idx_to_item = {i: m for m, i in self.item_to_idx.items()}
        log.info("Users: %d  |  Items: %d  |  Embedding dim: %d", self.num_users, self.num_items, embedding_dim)

        # Initialize and load model
        self.model = MatrixFactorizationWithBias(
            self.num_users, self.num_items, embedding_dim
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        log.info("Model loaded successfully ✓")

    def ensure_ratings_df_loaded(self):
        """Load ratings.parquet only when a heavy endpoint needs it."""
        if self.ratings_df is not None:
            return
        if not os.path.exists(RATINGS_PATH):
            raise RuntimeError(f"ratings.parquet not found at {RATINGS_PATH}")
        log.info("Loading ratings.parquet for /api/movies/popular …")
        self.ratings_df = pd.read_parquet(RATINGS_PATH)

    # ------------------------------------------------------------------ #
    def recommend(self, movie_ids: List[int], k: int = TOP_K_DEFAULT):
        """
        Given a list of watched movieIds, return top-k recommendations.

        Strategy: average the item embeddings of watched movies to form a
        synthetic user vector, then rank all other items by dot-product +
        item bias + global bias.

        Returns a list of dicts: [{id, title, rating}, …]
        """
        valid_ids   = [m for m in movie_ids if m in self.item_to_idx]
        invalid_ids = [m for m in movie_ids if m not in self.item_to_idx]

        if invalid_ids:
            log.warning("Unknown movieIds (ignored): %s", invalid_ids)

        if not valid_ids:
            raise HTTPException(
                status_code=422,
                detail=f"None of the provided movieIDs exist in the dataset: {movie_ids}",
            )

        with torch.no_grad():
            watched_idx = torch.tensor(
                [self.item_to_idx[m] for m in valid_ids],
                device=self.device,
                dtype=torch.long,
            )
            # Synthetic user = mean of watched item embeddings
            user_vec = self.model.item_embeddings(watched_idx).mean(dim=0)  # (D,)

            # Score every item
            # Use model weights directly (avoids duplicating large embedding tensors in RAM).
            scores = (self.model.item_embeddings.weight @ user_vec).detach().cpu().numpy()  # (N,)
            scores += self.model.item_bias.weight.detach().squeeze(-1).cpu().numpy()
            scores += float(self.model.global_bias.detach().cpu().item())

        # Exclude watched items
        watched_set = set(valid_ids)

        candidates = [
            {"movieId": self.idx_to_item[i], "score": float(scores[i])}
            for i in range(self.num_items)
            if self.idx_to_item[i] not in watched_set
        ]

        # Sort and take top k
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top = candidates[:k]

        # Attach titles
        id_to_title = (
            self.movies_df.set_index("movieId")["title"].to_dict()
        )
        result = []
        for c in top:
            mid = c["movieId"]
            result.append(
                {
                    "id":     str(mid),
                    "title":  id_to_title.get(mid, "Unknown"),
                    "rating": round(c["score"], 3),
                }
            )

        log.info(
            "Recommended %d movies for input %s  (top: %s)",
            len(result),
            valid_ids,
            result[0]["title"] if result else "—",
        )
        return result


# Pydantic schemas
class RecommendRequest(BaseModel):
    movieIDs: List[str]      # frontend sends strings ("79132", "6", …)
    k: int = TOP_K_DEFAULT   # optional: number of results

    @field_validator("movieIDs")
    @classmethod
    def not_empty(cls, v):
        if not v:
            raise ValueError("movieIDs list must not be empty")
        return v


class MovieOut(BaseModel):
    id:     str
    title:  str
    rating: float


class RecommendResponse(BaseModel):
    recommendations: List[MovieOut]
    valid_input_count: int
    requested_k: int


# App + lifespan (startup / shutdown)
recommender: MovieRecommender | None = None   # global singleton
recommender_lock = threading.Lock()

def get_recommender() -> MovieRecommender:
    """
    Lazy-load the heavy recommender so the service can start even under tight
    memory limits (e.g., free-tier deployments).
    """
    global recommender
    if recommender is None:
        with recommender_lock:
            if recommender is None:
                log.info("Loading recommender on first request…")
                recommender = MovieRecommender()
    return recommender


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    log.info("=== CineMind API starting up ===")

    # Only MODEL_PATH and MOVIES_PATH are required; avoid loading heavy tensors here.
    missing = [p for p in [MODEL_PATH, MOVIES_PATH] if not os.path.exists(p)]
    if missing:
        log.error("Missing required files: %s", missing)
        log.error("Make sure model_weights.pth and movies.csv are present.")
        raise RuntimeError(f"Missing required files: {missing}")
    
    # RATINGS_PATH is optional (only needed for legacy models or /api/movies/popular endpoint)
    if not os.path.exists(RATINGS_PATH):
        log.warning("ratings.parquet not found. /api/movies/popular endpoint will be unavailable.")

    log.info("=== API ready ===")
    yield
    # shutdown
    log.info("=== CineFind API shutting down ===")


app = FastAPI(
    title="CineFind Movie Recommendation API",
    version="1.0.0",
    description="Matrix-factorization recommender backed by a PyTorch model.",
    lifespan=lifespan,
)

# Allow the frontend (port 8080) to call the API (port 5000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# Routes
@app.api_route("/", methods=["GET", "HEAD"], tags=["health"])
def root():
    """Serve the frontend UI."""
    index_path = os.path.join(BASE_DIR, "frontend", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    # Fallback for dev/debug if frontend isn't present.
    return {"status": "ok", "message": "Frontend not found; API is running 🎬"}


@app.get("/health", tags=["health"])
def health():
    """Returns whether the model is loaded and ready."""
    loaded = recommender is not None
    return {
        "status": "ready" if loaded else "not_ready",
        "model_loaded": loaded,
        "device": recommender.device if loaded else None,
        "num_items": recommender.num_items if loaded else None,
    }

@app.get("/movies_1.json", include_in_schema=False)
def movies_1_json():
    """
    Serves movie titles to the frontend.
    The JS fetches it as a relative URL (e.g. `fetch('movies_1.json')`).
    """
    json_path = os.path.join(BASE_DIR, "data", "processed", "movies_1.json")
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="movies_1.json not found.")
    return FileResponse(json_path)


@app.post(
    "/api/movies/recommendations",
    response_model=RecommendResponse,
    tags=["recommendations"],
)
def get_recommendations(body: RecommendRequest):
    """
    Receive a list of movieIDs the user liked and return top-k recommendations.

    Request body:
        {
            "movieIDs": ["6", "79132", "103688"],
            "k": 10          // optional, default 10
        }

    Response:
        {
            "recommendations": [
                {"id": "318", "title": "Shawshank Redemption, The (1994)", "rating": 4.43},
                ...
            ],
            "valid_input_count": 3,
            "requested_k": 10
        }
    """
    rec = get_recommender()

    # Convert string IDs → int (frontend sends strings)
    try:
        int_ids = [int(mid) for mid in body.movieIDs]
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Non-integer movieID: {e}")

    k = max(1, min(body.k, 100))   # clamp between 1 and 100

    recs = rec.recommend(int_ids, k=k)

    valid_count = len([m for m in int_ids if m in rec.item_to_idx])

    return RecommendResponse(
        recommendations=recs,
        valid_input_count=valid_count,
        requested_k=k,
    )


@app.get("/api/movies/popular", tags=["movies"])
def popular_movies(n: int = 20):
    """Return the n most-rated movies (useful for debugging / seeding the UI)."""
    if not os.path.exists(RATINGS_PATH):
        raise HTTPException(
            status_code=503,
            detail="ratings.parquet not found; /api/movies/popular is unavailable.",
        )

    rec = get_recommender()
    try:
        rec.ensure_ratings_df_loaded()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    counts = (
        rec.ratings_df.groupby("movieId")
        .size()
        .reset_index(name="num_ratings")
        .merge(rec.movies_df[["movieId", "title"]], on="movieId", how="left")
        .sort_values("num_ratings", ascending=False)
        .head(n)
    )
    return counts[["movieId", "title", "num_ratings"]].to_dict(orient="records")


# Serve frontend static assets (styles.css, script.js, etc.)
# Keep this after API routes so `/api/*` and `/health` continue to work.
frontend_dir = os.path.join(BASE_DIR, "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir), name="frontend")