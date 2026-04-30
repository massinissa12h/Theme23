"""
recommender.py — All recommendation algorithms.

Engines:
  ContentRecommender       → TF-IDF cosine similarity on rich_text (memory-efficient)
  CollaborativeRecommender → mean-centered user-user similarity (raw scores, no normalisation)
  PopularityRecommender    → Bayesian popularity score (cold-start fallback)
  HybridRecommender        → weighted combination, normalises all scores at the end
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler



class ColouredFormatter(logging.Formatter):
    COLOURS = {
        logging.DEBUG:    "\033[36m",
        logging.INFO:     "\033[32m",
        logging.WARNING:  "\033[33m",
        logging.ERROR:    "\033[31m",
        logging.CRITICAL: "\033[1;31m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        record = logging.makeLogRecord(record.__dict__)
        colour = self.COLOURS.get(record.levelno, self.RESET)
        record.levelname = f"{colour}{record.levelname}{self.RESET}"
        return super().format(record)


console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setFormatter(ColouredFormatter("%(levelname)s | %(message)s"))
file_handler = logging.FileHandler("recommender.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler])
logger = logging.getLogger(__name__)



@dataclass
class RecommenderConfig:
    top_k_users: int = 2

    min_ratings_per_user: int = 2

    weight_content: float = 0.9
    weight_collaborative: float = 0.1

    default_n: int = 3

    def __post_init__(self):
        if abs(self.weight_content + self.weight_collaborative - 1.0) > 1e-6:
            raise ValueError("weight_content and weight_collaborative must sum to 1.0")
        if self.top_k_users < 1:
            raise ValueError("top_k_users must be at least 1")
        if self.min_ratings_per_user < 1:
            raise ValueError("min_ratings_per_user must be at least 1")
        if self.default_n < 1:
            raise ValueError("default_n must be at least 1")



def validate_products(products: list[dict]) -> None:
    """Make sure every product has required fields and no duplicate IDs."""
    if not products:
        raise ValueError("Products list must not be empty.")
    required = {"id", "name", "rich_text"}
    for i, p in enumerate(products):
        missing = required - p.keys()
        if missing:
            raise ValueError(f"Product at index {i} is missing fields: {missing}")
    ids = [p["id"] for p in products]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate product IDs found.")


def validate_ratings(ratings: pd.DataFrame) -> None:
    """Make sure ratings DataFrame has required columns and numeric ratings."""
    if ratings.empty:
        raise ValueError("Ratings DataFrame must not be empty.")
    required = {"user_id", "item_id", "rating"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"Ratings DataFrame is missing columns: {missing}")
    if not pd.api.types.is_numeric_dtype(ratings["rating"]):
        raise ValueError("Rating column must be numeric.")



class IndexMapper:
    """
    Converts UUID strings to integer indices and back.

    sklearn and numpy only work with integers.
    Supabase IDs are UUIDs like "00000000-0000-0000-0001-000000000001".

    Example:
        mapper = IndexMapper(["uuid-a", "uuid-b", "uuid-c"])
        mapper.to_int("uuid-b")  → 1
        mapper.to_uuid(1)        → "uuid-b"
    """

    def __init__(self, ids: list[str]):
        self._to_int:  dict[str, int] = {uid: i for i, uid in enumerate(ids)}
        self._to_uuid: dict[int, str] = {i: uid for uid, i in self._to_int.items()}

    def to_int(self, uid: str) -> Optional[int]:
        return self._to_int.get(uid)

    def to_uuid(self, idx: int) -> Optional[str]:
        return self._to_uuid.get(idx)

    def __len__(self) -> int:
        return len(self._to_int)



@dataclass
class Recommendation:
    """
    A single recommendation returned by any engine.

    sources shows how much each engine contributed:
        {"content": 0.82}                         → content only
        {"collaborative": 3.0}                    → collaborative only (raw score)
        {"content": 0.82, "collaborative": 0.31}  → both engines combined
        {"popularity": 0.75}                      → cold-start fallback
    """
    item_id: str
    name:    str
    score:   float
    sources: dict = field(default_factory=dict)

    def __repr__(self):
        src = ", ".join(f"{k}={v:.3f}" for k, v in self.sources.items())
        return f"Recommendation(id={self.item_id}, name={self.name!r}, score={self.score:.3f}, [{src}])"



class PopularityRecommender:
    """
    Returns globally popular items. Used when other engines cannot run.

    Ranking formula:
        score = mean_rating x log(1 + interaction_count)

    Rewards products that are both highly rated AND frequently interacted with.
    A product with 1 purchase ranks lower than one with 50 purchases
    even if both have 5 stars.
    """

    def __init__(self, ratings: pd.DataFrame, products: pd.DataFrame):
        self.products = products.set_index("id")

        if ratings.empty:
            self.popularity = pd.DataFrame(
                {"score": [1.0] * len(products)},
                index=products["id"],
            )
            return

        popularity = (
            ratings.groupby("item_id")["rating"]
            .agg(count="count", mean="mean")
            .assign(score=lambda d: d["mean"] * np.log1p(d["count"]))
        )
        scaler = MinMaxScaler()
        popularity["score"] = scaler.fit_transform(popularity[["score"]])
        self.popularity = popularity.sort_values("score", ascending=False)

    def recommend(self, exclude_ids: Optional[list[str]] = None, n: int = 3) -> list[Recommendation]:
        exclude_ids = set(exclude_ids or [])
        results = []
        for item_id, row in self.popularity.iterrows():
            if item_id in exclude_ids:
                continue
            name = self.products.at[item_id, "name"] if item_id in self.products.index else str(item_id)
            results.append(Recommendation(
                item_id=str(item_id),
                name=name,
                score=float(row["score"]),
                sources={"popularity": float(row["score"])},
            ))
            if len(results) >= n:
                break
        return results



class ContentRecommender:
    """
    Recommends products similar to the one being viewed.
    Uses TF-IDF on rich_text (name + description + category + tags).

    Memory-efficient: stores only the sparse TF-IDF matrix,
    computes similarity for one item at a time on demand.

    Only needs item_id — does not care about who the user is.
    """

    def __init__(self, products: list[dict]):
        self.df = pd.DataFrame(products).set_index("id")

        self.vectorizer  = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["rich_text"])

        self.item_to_idx = {uid: i for i, uid in enumerate(self.df.index)}

    def recommend(self, item_id: str, n: int = 3) -> list[Recommendation]:
        idx = self.item_to_idx.get(item_id)
        if idx is None:
            logger.warning("Content: unknown item_id=%s — cold-start triggered.", item_id)
            return []

        sim_scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        sim_scores[idx] = 0.0

        top_indices = sim_scores.argsort()[::-1]
        top_indices = top_indices[top_indices != idx][:n]

        return [
            Recommendation(
                item_id=str(self.df.index[i]),
                name=self.df.iloc[i]["name"],
                score=float(sim_scores[i]),
                sources={"content": float(sim_scores[i])},
            )
            for i in top_indices if sim_scores[i] > 0
        ]



class _BaseCollaborativeRecommender:
    """
    Base class shared by CollaborativeRecommender and _NullCollaborativeRecommender.
    Fixes PyCharm type warnings in HybridRecommender.
    """
    products: pd.DataFrame = pd.DataFrame(
        columns=["name"]
    ).set_index(pd.Index([], name="id"))

    def recommend(self, user_id: str, n: int = 3) -> list[Recommendation]:
        raise NotImplementedError



class CollaborativeRecommender(_BaseCollaborativeRecommender):
    """
    Recommends products liked by users similar to the current user.

    Returns RAW predicted scores — no normalisation here.
    Normalisation happens in:
        HybridRecommender.recommend() → when combining with content scores
        /homepage endpoint            → when calling collaborative directly

    Steps:
        1. Build user x product ratings matrix from interactions
        2. Mean-center each user's ratings (corrects for biased raters)
        3. Compute cosine similarity between all users
        4. Find top K most similar neighbours
        5. Predict ratings using weighted average of neighbour ratings
        6. Exclude products the user already interacted with
        7. Return highest predicted scores (raw, not normalised)
    """

    def __init__(self, ratings: pd.DataFrame, products: pd.DataFrame, config: RecommenderConfig):
        self.products    = products.set_index("id")
        self.top_k_users = config.top_k_users

        self.user_mapper = IndexMapper(ratings["user_id"].unique().tolist())
        self.item_mapper = IndexMapper(ratings["item_id"].unique().tolist())

        r = ratings.copy()
        r["user_idx"] = r["user_id"].map(self.user_mapper.to_int)
        r["item_idx"] = r["item_id"].map(self.item_mapper.to_int)

        full_matrix = (
            r.pivot_table(index="user_idx", columns="item_idx", values="rating")
            .fillna(0)
        )

        valid_users = full_matrix[
            full_matrix.gt(0).sum(axis=1) >= config.min_ratings_per_user
        ].index

        if valid_users.empty:
            raise ValueError(
                f"No users have at least {config.min_ratings_per_user} interactions. "
                "Lower min_ratings_per_user in RecommenderConfig."
            )
        self.matrix = full_matrix.loc[valid_users]

        user_means      = self.matrix.replace(0, np.nan).mean(axis=1)
        matrix_centered = self.matrix.subtract(user_means, axis=0).fillna(0)

        sim = cosine_similarity(matrix_centered)
        self.user_sim = pd.DataFrame(
            sim,
            index=self.matrix.index,
            columns=self.matrix.index,
        )

    def recommend(self, user_id: str, n: int = 3) -> list[Recommendation]:
        user_idx = self.user_mapper.to_int(user_id)

        if user_idx is None or user_idx not in self.user_sim.index:
            logger.warning("Collaborative: unknown user_id=%s — cold-start triggered.", user_id)
            return []

        neighbours = (
            self.user_sim[user_idx]
            .drop(index=user_idx, errors="ignore")
            .sort_values(ascending=False)
            .head(self.top_k_users)
        )

        weighted_sum = pd.Series(0.0, index=self.matrix.columns)
        sim_sum      = pd.Series(0.0, index=self.matrix.columns)
        for neighbour_idx, sim_score in neighbours.items():
            weighted_sum += self.matrix.loc[neighbour_idx] * sim_score
            sim_sum      += (self.matrix.loc[neighbour_idx] > 0).astype(float) * sim_score

        predicted = (weighted_sum / sim_sum.replace(0, np.nan)).dropna()

        rated_row     = self.matrix.loc[user_idx]
        already_rated = rated_row[rated_row > 0].index
        predicted     = predicted.drop(index=already_rated, errors="ignore")

        predicted = predicted.sort_values(ascending=False).head(n)

        results = []
        for item_idx, score in predicted.items():
            item_uuid = self.item_mapper.to_uuid(item_idx)
            if item_uuid is None:
                continue
            name = self.products.at[item_uuid, "name"] if item_uuid in self.products.index else item_uuid
            results.append(Recommendation(
                item_id=item_uuid,
                name=name,
                score=float(score),
                sources={"collaborative": float(score)},
            ))
        return results



class _NullCollaborativeRecommender(_BaseCollaborativeRecommender):
    """
    Stand-in when not enough data to build collaborative engine.
    Always returns empty list → triggers cold-start in hybrid.
    """

    def recommend(self, user_id: str, n: int = 3) -> list[Recommendation]:
        return []



class HybridRecommender:
    """
    Combines all three engines into one final ranked list.

    Flow:
        1. Get collaborative recs for user_id  (raw scores)
        2. Get content recs for item_id        (cosine similarity scores)
        3. Cold-start: unknown user → swap collaborative with popularity
        4. Cold-start: unknown item → swap content with popularity
        5. Apply weights: content × 0.9 + collaborative × 0.1
        6. Merge all scores into one list
        7. Normalise final combined scores to [0, 1]
        8. Sort descending, remove viewed item, return top N
    """

    def __init__(
        self,
        content:       ContentRecommender,
        collaborative: _BaseCollaborativeRecommender,
        popularity:    PopularityRecommender,
        config:        RecommenderConfig,
    ):
        self.content       = content
        self.collaborative = collaborative
        self.popularity    = popularity
        self.w_content     = config.weight_content
        self.w_collab      = config.weight_collaborative

    def recommend(self, user_id: str, item_id: str, n: int = 3) -> list[Recommendation]:
        collab_recs  = self.collaborative.recommend(user_id, n=n * 2)
        content_recs = self.content.recommend(item_id, n=n * 2)

        cold_start_user = not collab_recs
        cold_start_item = not content_recs

        if cold_start_user:
            logger.info("Cold-start: unknown user_id=%s → using popularity.", user_id)
            collab_recs = self.popularity.recommend(n=n * 2)

        if cold_start_item:
            logger.info("Cold-start: unknown item_id=%s → using popularity.", item_id)
            content_recs = self.popularity.recommend(n=n * 2)

        score_map: dict[str, dict] = {}

        for rec in collab_recs:
            weighted = rec.score * (1.0 if cold_start_user else self.w_collab)
            if weighted > 0:
                score_map.setdefault(rec.item_id, {})["collaborative"] = weighted

        for rec in content_recs:
            weighted = rec.score * (1.0 if cold_start_item else self.w_content)
            if weighted > 0:
                score_map.setdefault(rec.item_id, {})["content"] = weighted

        combined: list[Recommendation] = []
        for iid, srcs in score_map.items():
            combined.append(Recommendation(
                item_id=iid,
                name=self._lookup_name(iid),
                score=sum(srcs.values()),
                sources=srcs,
            ))

        if not combined:
            logger.warning(
                "No recommendations for user=%s, item=%s → full popularity fallback.",
                user_id, item_id,
            )
            return self.popularity.recommend(n=n)

        if len(combined) > 1:
            max_s = max(r.score for r in combined)
            min_s = min(r.score for r in combined)
            for r in combined:
                r.score = (r.score - min_s) / (max_s - min_s) if max_s > min_s else 1.0

        combined.sort(key=lambda r: r.score, reverse=True)

        if not cold_start_user:
            combined = [r for r in combined if r.item_id != item_id]

        return combined[:n]

    def _lookup_name(self, item_id: str) -> str:
        """Get product name by UUID, return UUID string if not found."""
        try:
            return self.content.df.at[item_id, "name"]
        except KeyError:
            return item_id



def precision_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    """Fraction of top-K recommendations that are actually relevant."""
    top_k = recommended[:k]
    return len(set(top_k) & set(relevant)) / k if k else 0.0


def recall_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    """Fraction of all relevant items that appear in top-K recommendations."""
    top_k = recommended[:k]
    return len(set(top_k) & set(relevant)) / len(relevant) if relevant else 0.0



def build_engines(
    products: list[dict],
    ratings:  pd.DataFrame,
    config:   RecommenderConfig,
) -> HybridRecommender:
    """
    Validate inputs, build all engines, return a ready HybridRecommender.

    Handles missing data gracefully:
        No products     → raises ValueError
        No interactions → content + popularity only
        Too few ratings → collaborative skipped, logs warning
    """
    if not products:
        raise ValueError("Cannot build engines: no products available.")

    validate_products(products)
    products_df = pd.DataFrame(products)

    content_rec    = ContentRecommender(products)
    popularity_rec = PopularityRecommender(ratings, products_df)

    collab_rec: _BaseCollaborativeRecommender
    if not ratings.empty:
        validate_ratings(ratings)
        try:
            collab_rec = CollaborativeRecommender(ratings, products_df, config)
        except ValueError as exc:
            logger.warning("Collaborative engine disabled: %s", exc)
            collab_rec = _NullCollaborativeRecommender()
    else:
        logger.warning("No ratings — collaborative filtering disabled.")
        collab_rec = _NullCollaborativeRecommender()

    return HybridRecommender(content_rec, collab_rec, popularity_rec, config)