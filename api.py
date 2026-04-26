from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

import database as db
import recommender as rec

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Global state
# ──────────────────────────────────────────────

_state: dict = {
    "hybrid":  None,   # HybridRecommender instance
    "client":  None,   # Supabase client
}

CONFIG = rec.RecommenderConfig(
    top_k_users=2,
    min_ratings_per_user=2,
    weight_content=0.9,       # ← increased from 0.8
    weight_collaborative=0.1, # ← decreased from 0.2
    default_n=3,
)


def _retrain() -> None:
    """Fetch fresh data from Supabase and rebuild all engines."""
    client   = _state["client"]
    products = db.fetch_products(client)
    ratings  = db.fetch_ratings(client)
    _state["hybrid"] = rec.build_engines(products, ratings, CONFIG)
    logger.info("Model retrained successfully.")


# ──────────────────────────────────────────────
# App lifespan (startup / shutdown)
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — connecting to Supabase and training model...")
    _state["client"] = db.get_client()
    _retrain()
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Hybrid Recommender API", lifespan=lifespan)


# ──────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────

class InteractionIn(BaseModel):
    user_id:    str
    product_id: str
    action:     str   # "view" | "add_to_cart" | "purchase"


class RecommendationOut(BaseModel):
    item_id: str
    name:    str
    score:   float
    sources: dict


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _state["hybrid"] is not None}


@app.get("/recommend", response_model=list[RecommendationOut])
def recommend(
    user_id: str = Query(..., description="UUID of the user"),
    item_id: str = Query(..., description="UUID of the product being viewed"),
    n: int       = Query(3,   description="Number of recommendations to return"),
):
    client = _state["client"]
    hybrid = _state["hybrid"]

    if hybrid is None:
        raise HTTPException(status_code=503, detail="Model not ready yet.")

    # Always run the engine — cache is for storage only, not skipping the model
    results = hybrid.recommend(user_id=user_id, item_id=item_id, n=n)

    # Save to cache
    db.upsert_recommendations(
        client, user_id, item_id,
        [{"recommended": r.item_id, "score": r.score, "sources": r.sources} for r in results],
    )

    return [
        RecommendationOut(item_id=r.item_id, name=r.name, score=r.score, sources=r.sources)
        for r in results
    ]

@app.post("/ratings", status_code=201)
def add_interaction(interaction: InteractionIn):
    client = _state["client"]

    if interaction.action not in db.INTERACTION_WEIGHTS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown action '{interaction.action}'. "
                   f"Valid actions: {list(db.INTERACTION_WEIGHTS.keys())}",
        )

    # Save interaction to Supabase
    client.table("interactions").insert({
        "user_id":    interaction.user_id,
        "product_id": interaction.product_id,
        "action":     interaction.action,
    }).execute()

    # Retrain so the new signal is reflected immediately
    try:
        _retrain()
    except Exception as exc:
        logger.error("Retrain failed after new interaction: %s", exc)

    return {"detail": "Interaction recorded and model retrained."}


@app.post("/retrain", status_code=200)
def retrain():
    """Force a full retrain from Supabase data (admin use)."""
    try:
        _retrain()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"detail": "Model retrained successfully."}