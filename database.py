

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
logger = logging.getLogger(__name__)

INTERACTION_WEIGHTS: dict[str, float] = {
    "seen":     1.0,   # user viewed a product page
    "share":    1.5,   # user shared a product
    "save":     2.0,   # user saved/wishlisted a product
    "cart":     3.0,   # user added to cart
    "purchase": 5.0,   # user completed a purchase
}


def get_client(url: Optional[str] = None, key: Optional[str] = None) -> Client:
    """Create Supabase client. Uses env vars or explicit credentials."""
    url = url or os.environ.get("SUPABASE_URL")
    key = key or os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")
    return create_client(url, key)


def fetch_products(client: Client, batch_size: int = 1000) -> list[dict]:
    """
    Fetch all products with tags via single join.
    Automatically paginates for large catalogs.
    """
    all_products: list[dict] = []
    offset = 0

    while True:
        try:
            resp = (
                client.table("products")
                .select("id, name, description, category, product_tags(tag)")
                .range(offset, offset + batch_size - 1)
                .execute()
            )
            batch = resp.data
        except Exception as exc:
            logger.error("Failed to fetch products (offset %d): %s", offset, exc)
            break

        if not batch:
            break

        for p in batch:
            tags = [t["tag"] for t in p.pop("product_tags", [])]
            parts = [
                p.get("name") or "",
                p.get("description") or "",
                p.get("category") or "",
            ]
            p["rich_text"] = " ".join(filter(None, parts + tags))

        all_products.extend(batch)
        offset += batch_size

        if len(batch) < batch_size:
            break

    if not all_products:
        logger.warning("Products table is empty — no recommendations possible.")
    else:
        logger.info("Fetched %d products from Supabase.", len(all_products))

    return all_products


def fetch_ratings(client: Client) -> pd.DataFrame:
    """Convert raw interactions into synthetic ratings, keeping only the strongest action."""
    empty = pd.DataFrame(columns=["user_id", "item_id", "rating"])

    try:
        resp = client.table("interactions").select("user_id, product_id, action").execute()
        rows = resp.data
    except Exception as exc:
        logger.error("Failed to fetch interactions: %s", exc)
        return empty

    if not rows:
        logger.warning("Interactions table is empty — collaborative filtering disabled.")
        return empty

    df = pd.DataFrame(rows).rename(columns={"product_id": "item_id"})
    df["rating"] = df["action"].map(INTERACTION_WEIGHTS)
    df = df.dropna(subset=["rating"])

    if df.empty:
        logger.warning("No recognised interaction actions found.")
        return empty

    df = df.sort_values("rating", ascending=False).drop_duplicates(
        subset=["user_id", "item_id"], keep="first"
    )
    return df[["user_id", "item_id", "rating"]].reset_index(drop=True)


def upsert_recommendations(
    client: Client,
    user_id: str,
    item_id: str,
    recommendations: list[dict],
) -> None:
    """
    Save recommendations to cache using atomic upsert.
    Requires a UNIQUE constraint on (user_id, item_id, recommended) in Supabase.
    """
    if not recommendations:
        return

    rows = [
        {
            "user_id": user_id,
            "item_id": item_id,
            "recommended": r["recommended"],
            "score": r["score"],
            "sources": r["sources"],
        }
        for r in recommendations
    ]

    try:
        client.table("recommendations").upsert(rows).execute()
        logger.info(
            "Cached %d recommendations for user=%s item=%s.",
            len(rows), user_id, item_id,
        )
    except Exception as exc:
        logger.error("Cache upsert failed: %s", exc)


def fetch_cached_recommendations(
    client: Client,
    user_id: str,
    item_id: str,
) -> list[dict] | None:
    """Retrieve cached recommendations ordered by score descending."""
    try:
        resp = (
            client.table("recommendations")
            .select("recommended, score, sources")
            .eq("user_id", user_id)
            .eq("item_id", item_id)
            .order("score", desc=True)
            .execute()
        )
        return resp.data if resp.data else None
    except Exception as exc:
        logger.error("Failed to fetch cached recommendations: %s", exc)
        return None