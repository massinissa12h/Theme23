"""
database.py — Supabase data access layer.

Responsibilities:
  - Fetch products (with tags and category merged into rich_text)
  - Fetch interactions and convert to synthetic ratings
  - Read and write the recommendations cache
"""

from __future__ import annotations

import logging
import os

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
logger = logging.getLogger(__name__)

# ── Interaction → synthetic rating weights ────────────────────────────────────
# We have no ratings table — interactions are converted to synthetic ratings.
# The stronger the action, the higher the weight.
INTERACTION_WEIGHTS: dict[str, float] = {
    "view":          1.0,   # weakest signal
    "add_to_cart":   3.0,   # medium signal
    "purchase":      5.0,   # strongest signal
}


def get_client() -> Client:
    """Create and return a Supabase client using credentials from .env"""
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    return create_client(url, key)


# ── Products ──────────────────────────────────────────────────────────────────

def fetch_products(client: Client) -> list[dict]:
    """
    Fetch all products from Supabase and build rich_text for each one.

    rich_text = name + description + category + tags merged into one string.
    This is what the TF-IDF content engine uses for similarity.

    Example:
        name      = "Nike Running Shoes"
        rich_text = "Nike Running Shoes Lightweight running shoes Footwear running sport nike"
                     ^ name              ^ description              ^ category ^ tags
    """
    try:
        products_resp = client.table("products").select("id, name, description, category").execute()
        products: list[dict] = products_resp.data
    except Exception as exc:
        logger.error("Failed to fetch products from Supabase: %s", exc)
        return []

    if not products:
        logger.warning("Products table is empty — no recommendations possible.")
        return []

    # Fetch all tags grouped by product_id
    try:
        tags_resp = client.table("product_tags").select("product_id, tag").execute()
        tags_by_product: dict[str, list[str]] = {}
        for row in tags_resp.data:
            tags_by_product.setdefault(row["product_id"], []).append(row["tag"])
    except Exception as exc:
        logger.warning("Failed to fetch product tags (continuing without them): %s", exc)
        tags_by_product = {}

    # Build rich_text for each product by merging all text fields
    for p in products:
        parts = [
            p.get("name")        or "",
            p.get("description") or "",
            p.get("category")    or "",
        ]
        parts += tags_by_product.get(p["id"], [])
        p["rich_text"] = " ".join(filter(None, parts))

    logger.info("Fetched %d products from Supabase.", len(products))
    return products


# ── Ratings (derived from interactions) ───────────────────────────────────────

def fetch_ratings(client: Client) -> pd.DataFrame:
    """
    Convert raw interaction rows into a synthetic ratings DataFrame.

    Since we have no ratings table, we derive ratings from interactions
    using INTERACTION_WEIGHTS:
        view        → 1.0
        add_to_cart → 3.0
        purchase    → 5.0

    For each (user, product) pair we keep only the STRONGEST action.
    This prevents 5 views from being treated the same as a purchase.

    Example:
        user views p1 five times       → rating = 1.0  (still just a view)
        user views then purchases p1   → rating = 5.0  (purchase wins)
        user views then carts p1       → rating = 3.0  (add_to_cart wins)

    Returns DataFrame with columns: user_id, item_id, rating
    """
    empty = pd.DataFrame(columns=["user_id", "item_id", "rating"])

    try:
        resp = client.table("interactions").select("user_id, product_id, action").execute()
        rows = resp.data
    except Exception as exc:
        logger.error("Failed to fetch interactions from Supabase: %s", exc)
        return empty

    if not rows:
        logger.warning("Interactions table is empty — collaborative filtering disabled.")
        return empty

    df = pd.DataFrame(rows).rename(columns={"product_id": "item_id"})

    # Drop unknown action types not in INTERACTION_WEIGHTS
    df["rating"] = df["action"].map(INTERACTION_WEIGHTS)
    df = df.dropna(subset=["rating"])

    if df.empty:
        logger.warning("No recognised interaction actions found — check INTERACTION_WEIGHTS.")
        return empty

    # Priority ranking: higher = stronger signal
    # Used to pick the winning action per (user, item)
    ACTION_PRIORITY = {
        "view":        1,
        "add_to_cart": 2,
        "purchase":    3,
    }
    df["priority"] = df["action"].map(ACTION_PRIORITY)

    # Sort by priority descending so the strongest action comes first
    df = df.sort_values("priority", ascending=False)

    # Keep only the strongest action per (user, item) pair
    df = df.drop_duplicates(subset=["user_id", "item_id"], keep="first")

    # Final ratings — just user_id, item_id, rating
    ratings = df[["user_id", "item_id", "rating"]].reset_index(drop=True)

    logger.info(
        "Derived %d synthetic ratings from %d interactions.",
        len(ratings), len(rows),
    )
    return ratings


# ── Recommendations cache ──────────────────────────────────────────────────────

def upsert_recommendations(
    client: Client,
    user_id: str,
    item_id: str,
    recommendations: list[dict],
) -> None:
    """
    Save fresh recommendations to the cache table in Supabase.
    Deletes old entries for this (user_id, item_id) pair first.

    Each recommendation dict must have:
        recommended → UUID of the recommended product
        score       → final hybrid score (float)
        sources     → {"content": 0.82, "collaborative": 0.41}
    """
    # Clear stale cache for this user + viewed item
    client.table("recommendations").delete().match(
        {"user_id": user_id, "item_id": item_id}
    ).execute()

    if not recommendations:
        return

    rows = [
        {
            "user_id":     user_id,
            "item_id":     item_id,
            "recommended": r["recommended"],
            "score":       r["score"],
            "sources":     r["sources"],
        }
        for r in recommendations
    ]
    client.table("recommendations").insert(rows).execute()
    logger.info(
        "Cached %d recommendations for user=%s viewing item=%s.",
        len(rows), user_id, item_id,
    )


def fetch_cached_recommendations(
    client: Client,
    user_id: str,
    item_id: str,
) -> list[dict] | None:
    """
    Check if recommendations already exist in cache for this
    (user_id, item_id) pair. Returns None if cache is empty.
    """
    resp = (
        client.table("recommendations")
        .select("recommended, score, sources")
        .eq("user_id", user_id)
        .eq("item_id", item_id)
        .order("score", desc=True)
        .execute()
    )
    return resp.data if resp.data else None