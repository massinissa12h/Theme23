"""
main.py — Production-ready FastAPI service.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from typing import Annotated, Optional

import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import database as db
import recommender as rec

logger = logging.getLogger(__name__)

# ── Settings ──────────────────────────────────────────────────────────────────

class Settings:
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.api_key = os.getenv("API_KEY", "dev-key-change-me")
        self.retrain_cooldown_seconds = int(os.getenv("RETRAIN_COOLDOWN_SECONDS", "30"))
        self.rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))
        self.rate_limit_window_seconds = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
        self.max_recommendations = int(os.getenv("MAX_RECOMMENDATIONS", "50"))

        if not self.supabase_url or not self.supabase_key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")


settings = Settings()

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ── Rate limiter ──────────────────────────────────────────────────────────────

class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window = window_seconds
        self._history: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

    async def is_allowed(self, key: str) -> bool:
        now = time.time()
        async with self._lock:
            history = self._history.get(key, [])
            history = [t for t in history if now - t < self.window]
            if len(history) >= self.max_requests:
                self._history[key] = history
                return False
            history.append(now)
            self._history[key] = history
            return True


rate_limiter = SlidingWindowRateLimiter(
    settings.rate_limit_requests,
    settings.rate_limit_window_seconds,
)

# ── Auth ──────────────────────────────────────────────────────────────────────

async def verify_api_key(x_api_key: Annotated[str, Header(..., alias="X-API-Key")]):
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_api_key

# ── Global state ──────────────────────────────────────────────────────────────

_state: dict = {
    "hybrid":       None,
    "client":       None,
    "last_retrain": 0.0,
    "retrain_lock": asyncio.Lock(),
    "executor":     None,
}

CONFIG = rec.RecommenderConfig(
    top_k_users=2,
    min_ratings_per_user=2,
    weight_content=0.9,
    weight_collaborative=0.1,
    default_n=3,
)


def _build_engines_sync(products: list[dict], ratings_dict: list[dict]) -> rec.HybridRecommender:
    """Sync wrapper for process-pool serialization."""
    ratings = pd.DataFrame(ratings_dict) if ratings_dict else pd.DataFrame(
        columns=["user_id", "item_id", "rating"]
    )
    return rec.build_engines(products, ratings, CONFIG)


async def _retrain() -> None:
    """Debounced, locked retraining. CPU work offloaded to process pool."""
    async with _state["retrain_lock"]:
        now = time.time()
        if now - _state["last_retrain"] < settings.retrain_cooldown_seconds:
            logger.debug("Retrain skipped: within cooldown window.")
            return

        client = _state.get("client")
        if client is None:
            logger.warning("Retrain skipped: no Supabase client.")
            return

        try:
            logger.info("Starting model retrain...")
            products    = await asyncio.to_thread(db.fetch_products, client)
            ratings_df  = await asyncio.to_thread(db.fetch_ratings, client)
            ratings_dict = ratings_df.to_dict("records") if not ratings_df.empty else []

            loop   = asyncio.get_running_loop()
            hybrid = await loop.run_in_executor(
                _state["executor"],
                _build_engines_sync,
                products,
                ratings_dict,
            )

            _state["hybrid"]       = hybrid
            _state["last_retrain"] = time.time()
            logger.info("Model retrained successfully.")
        except Exception as exc:
            logger.error("Retrain failed: %s", exc, exc_info=True)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    _state["client"]   = db.get_client(settings.supabase_url, settings.supabase_key)
    _state["executor"] = ProcessPoolExecutor(max_workers=1)
    await _retrain()
    yield
    logger.info("Shutting down...")
    if _state["executor"]:
        _state["executor"].shutdown(wait=True)


app = FastAPI(
    title="Hybrid Recommender API",
    version="2.0.0",
    lifespan=lifespan,
)

# ── Middleware ───────────────────────────────────────────────────────────────

@app.middleware("http")
async def production_middleware(request: Request, call_next):
    if request.url.path != "/health":
        client_ip = request.client.host if request.client else "unknown"
        allowed = await rate_limiter.is_allowed(client_ip)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
            )

    start    = time.time()
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    logger.info(
        "%s %s — %d — %.2fms",
        request.method, request.url.path, response.status_code, duration,
    )
    return response


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(
        "Unhandled exception at %s %s: %s",
        request.method, request.url.path, exc, exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

# ── Schemas ───────────────────────────────────────────────────────────────────

class InteractionIn(BaseModel):
    user_id:    str
    product_id: str
    action:     str   # "view" | "add_to_cart" | "purchase"


class RecommendationOut(BaseModel):
    item_id: str
    name:    str
    score:   float
    sources: dict


class HomepageOut(BaseModel):
    item_id:      str
    name:         str
    score:        float
    sources:      dict
    personalised: bool  # True = based on user history, False = popularity fallback


class HealthOut(BaseModel):
    status:       str
    model_ready:  bool
    last_retrain: float | None
    uptime_hint:  str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthOut)
def health():
    last = _state.get("last_retrain")
    return {
        "status":       "ok",
        "model_ready":  _state["hybrid"] is not None,
        "last_retrain": last,
        "uptime_hint":  f"{int(time.time() - last)}s ago" if last else "never",
    }


@app.get("/homepage", response_model=list[HomepageOut], dependencies=[Depends(verify_api_key)])
async def homepage(
    n: int = Query(3, ge=1, le=settings.max_recommendations),
    user_id: Optional[str] = Query(None, description="UUID of the user (optional)"),
):
    """
    Returns recommendations for the homepage.

    Flow:
        No user_id provided  → pure popularity (new visitor)
        Known user_id        → collaborative picks from their history
        Unknown user_id      → pure popularity (new registered user, no history yet)

    Frontend uses personalised flag:
        personalised=false → show "Trending Products"
        personalised=true  → show "Picked For You"
    """
    hybrid = _state["hybrid"]

    if hybrid is None:
        raise HTTPException(status_code=503, detail="Model not ready yet.")

    # No user_id → pure popularity
    if user_id is None:
        recs = hybrid.popularity.recommend(n=n)
        return [
            HomepageOut(
                item_id=r.item_id,
                name=r.name,
                score=r.score,
                sources=r.sources,
                personalised=False,
            )
            for r in recs
        ]

    # Known user → collaborative picks (no item context)
    collab_recs = hybrid.collaborative.recommend(user_id=user_id, n=n)

    if collab_recs:
        # Normalise raw collaborative scores to [0, 1]
        max_s = max(r.score for r in collab_recs)
        min_s = min(r.score for r in collab_recs)
        for r in collab_recs:
            normalised = (r.score - min_s) / (max_s - min_s) if max_s > min_s else 1.0
            r.score = normalised
            r.sources = {"collaborative": normalised}

        # Filter out zero-score results
        collab_recs = [r for r in collab_recs if r.score > 0]

        # If we don't have enough results, fill remaining slots with popular items
        if len(collab_recs) < n:
            already_shown = [r.item_id for r in collab_recs]
            filler = hybrid.popularity.recommend(
                exclude_ids=already_shown,
                n=n - len(collab_recs),
            )
            collab_recs += filler

        return [
            HomepageOut(
                item_id=r.item_id,
                name=r.name,
                score=r.score,
                sources=r.sources,
                personalised=len(r.sources) > 0 and "popularity" not in r.sources,
            )
            for r in collab_recs
        ]

    # Unknown or new user → pure popularity
    logger.info("Homepage: unknown or new user_id=%s → popularity fallback.", user_id)
    recs = hybrid.popularity.recommend(n=n)
    return [
        HomepageOut(
            item_id=r.item_id,
            name=r.name,
            score=r.score,
            sources=r.sources,
            personalised=True,  # ✅ CORRIGÉ : marqué comme personnalisé si user_id est connu
        )
        for r in recs
    ]


@app.get("/recommend", response_model=list[RecommendationOut], dependencies=[Depends(verify_api_key)])
async def recommend(
    user_id: str = Query(..., description="UUID of the user"),
    item_id: str = Query(..., description="UUID of the product being viewed"),
    n: int = Query(3, ge=1, le=settings.max_recommendations),
    refresh: bool = Query(False, description="Bypass cache and recompute"),
):
    """
    Returns top-N recommendations for a user viewing a specific product.

    Flow:
        1. Check cache → return instantly if hit (unless refresh=true)
        2. Run hybrid engine
        3. Save results to cache
        4. Return results
    """
    client = _state["client"]
    hybrid = _state["hybrid"]

    if hybrid is None:
        raise HTTPException(status_code=503, detail="Model not ready yet.")

    # 1. Cache-first (skip if refresh=true)
    if not refresh:
        cached = await asyncio.to_thread(
            db.fetch_cached_recommendations, client, user_id, item_id
        )
        if cached:
            logger.info("Cache hit for user=%s item=%s", user_id, item_id)
            # ✅ CORRIGÉ : normalise les scores avant de les retourner
            results = [
                RecommendationOut(
                    item_id=r["recommended"],
                    name=hybrid._lookup_name(r["recommended"]),
                    score=r["score"],
                    sources=r["sources"],
                )
                for r in cached[:n]
            ]
            # Normalise les scores du cache
            if len(results) > 1:
                max_s = max(r.score for r in results)
                min_s = min(r.score for r in results)
                for r in results:
                    r.score = (r.score - min_s) / (max_s - min_s) if max_s > min_s else 1.0
            return results

    # 2. Run the hybrid engine
    results = hybrid.recommend(user_id=user_id, item_id=item_id, n=n)

    # 3. Save to cache asynchronously (before normalization)
    await asyncio.to_thread(
        db.upsert_recommendations,
        client,
        user_id,
        item_id,
        [{"recommended": r.item_id, "score": r.score, "sources": r.sources} for r in results],
    )

    # 4. Normalize scores before returning (to ensure consistency)
    if len(results) > 1:
        max_s = max(r.score for r in results)
        min_s = min(r.score for r in results)
        for r in results:
            r.score = (r.score - min_s) / (max_s - min_s) if max_s > min_s else 1.0

    # 5. Return results
    return [
        RecommendationOut(item_id=r.item_id, name=r.name, score=r.score, sources=r.sources)
        for r in results
    ]


def _insert_interaction_sync(client, user_id: str, product_id: str, action: str):
    """Synchronous insert wrapper for thread execution."""
    try:
        resp = client.table("interactions").insert({
            "user_id":    user_id,
            "product_id": product_id,
            "action":     action,
        }).execute()
        logger.info(
            "Inserted interaction: user=%s product=%s action=%s",
            user_id, product_id, action,
        )
        return resp
    except Exception as exc:
        logger.error("Supabase insert failed: %s", exc, exc_info=True)
        raise


@app.post("/ratings", status_code=202, dependencies=[Depends(verify_api_key)])
async def add_interaction(
    interaction: InteractionIn,
    background_tasks: BackgroundTasks,
):
    """
    Record a new user interaction and schedule a model retrain.

    Returns 202 immediately — retrain happens in the background
    so the user does not wait.
    """
    client = _state["client"]

    if interaction.action not in db.INTERACTION_WEIGHTS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown action '{interaction.action}'. "
                   f"Valid: {list(db.INTERACTION_WEIGHTS.keys())}",
        )

    # Insert interaction into Supabase
    try:
        await asyncio.to_thread(
            _insert_interaction_sync,
            client,
            interaction.user_id,
            interaction.product_id,
            interaction.action,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Database insert failed: {str(exc)}")

    # Schedule retrain in background (debounced — skipped if within cooldown)
    background_tasks.add_task(_retrain)

    return {"detail": "Interaction recorded. Model update scheduled."}


@app.post("/retrain", status_code=200, dependencies=[Depends(verify_api_key)])
async def manual_retrain():
    """Force a full model retrain from Supabase data (admin use)."""
    await _retrain()
    if _state["hybrid"] is None:
        raise HTTPException(status_code=500, detail="Retrain failed — check logs.")
    return {"detail": "Model retrained successfully."}