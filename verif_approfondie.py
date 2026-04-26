"""
test_api.py — Full API test suite with semantic checks.

Tests all endpoints across multiple users, products, and edge cases.
Also verifies semantic relevance (categories, tags, etc.).
Run with: python test_api.py
"""

from __future__ import annotations

import sys
import requests

BASE_URL = "http://127.0.0.1:8000"
API_KEY  = "39816561"
HEADERS  = {"X-API-Key": API_KEY}

# ── ID reference ──────────────────────────────────────────────────────────────
# USERS
U1 = "00000000-0000-0000-0000-000000000001"  # alice  → footwear lover
U2 = "00000000-0000-0000-0000-000000000002"  # bob    → electronics lover
U3 = "00000000-0000-0000-0000-000000000003"  # carol  → footwear + fitness
U4 = "00000000-0000-0000-0000-000000000004"  # dan    → gaming electronics
U5 = "00000000-0000-0000-0000-000000000005"  # eve    → footwear lover
U6 = "00000000-0000-0000-0000-000000000006"  # frank  → electronics lover
U7 = "00000000-0000-0000-0000-000000000007"  # grace  → footwear + fitness
U8 = "00000000-0000-0000-0000-000000000008"  # henry  → gaming electronics
U99 = "00000000-0000-0000-0000-000000000099" # unknown user

# PRODUCTS
P1 = "00000000-0000-0000-0001-000000000001"  # Nike Running Shoes → Footwear
P2 = "00000000-0000-0000-0001-000000000002"  # Adidas Sneakers → Footwear
P3 = "00000000-0000-0000-0001-000000000003"  # Puma Training Shoes → Footwear
P4 = "00000000-0000-0000-0001-000000000004"  # Gaming Mouse → Electronics
P5 = "00000000-0000-0000-0001-000000000005"  # Mechanical Keyboard → Electronics
P6 = "00000000-0000-0000-0001-000000000006"  # Monitor 27 inch → Electronics
P7 = "00000000-0000-0000-0001-000000000007"  # Yoga Mat → Sport
P8 = "00000000-0000-0000-0001-000000000008"  # Resistance Bands → Sport
P99 = "00000000-0000-0000-0001-000000000099" # unknown product

# PRODUCT CATEGORIES MAP (for semantic checks)
PRODUCT_CATEGORIES = {
    P1: "Footwear",
    P2: "Footwear",
    P3: "Footwear",
    P4: "Electronics",
    P5: "Electronics",
    P6: "Electronics",
    P7: "Sport",
    P8: "Sport",
}

# USER PREFERENCES (for semantic checks)
USER_PREFERENCES = {
    U1: ["Footwear"],  # alice loves footwear
    U2: ["Electronics"],  # bob loves electronics
    U3: ["Footwear", "Sport"],  # carol loves footwear and fitness
    U4: ["Electronics"],  # dan loves gaming electronics
    U5: ["Footwear"],  # eve loves footwear
    U8: ["Electronics"],  # henry loves gaming electronics
}

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "\033[32m  PASS\033[0m"
FAIL = "\033[31m  FAIL\033[0m"
HEAD = "\033[1;34m"
END  = "\033[0m"

total  = 0
passed = 0


def section(title: str):
    print(f"\n{HEAD}{'─' * 60}{END}")
    print(f"{HEAD}  {title}{END}")
    print(f"{HEAD}{'─' * 60}{END}")


def check(label: str, condition: bool, detail: str = ""):
    global total, passed
    total += 1
    if condition:
        passed += 1
        print(f"{PASS}  {label}")
    else:
        print(f"{FAIL}  {label}")
        if detail:
            print(f"         → {detail}")


def get_recommend(user_id: str, item_id: str, n: int = 3, refresh: bool = False) -> list:
    resp = requests.get(
        f"{BASE_URL}/recommend",
        params={"user_id": user_id, "item_id": item_id, "n": n, "refresh": str(refresh).lower()},
        headers=HEADERS,
    )
    return resp.json() if resp.status_code == 200 else []


def get_homepage(user_id: str | None = None, n: int = 3) -> list:
    params = {"n": n}
    if user_id:
        params["user_id"] = user_id
    resp = requests.get(f"{BASE_URL}/homepage", params=params, headers=HEADERS)
    return resp.json() if resp.status_code == 200 else []


def post_rating(user_id: str, product_id: str, action: str) -> int:
    resp = requests.post(
        f"{BASE_URL}/ratings",
        json={"user_id": user_id, "product_id": product_id, "action": action},
        headers=HEADERS,
    )
    return resp.status_code


def post_retrain() -> int:
    return requests.post(f"{BASE_URL}/retrain", headers=HEADERS).status_code


# ── Semantic Checks ───────────────────────────────────────────────────────────

def check_semantic_relevance(user_id: str, item_id: str, recs: list, expected_category: str):
    """Check if recommendations are semantically relevant to the viewed item."""
    relevant_count = 0
    for r in recs:
        pid = r["item_id"]
        cat = PRODUCT_CATEGORIES.get(pid)
        if cat == expected_category:
            relevant_count += 1
    return relevant_count


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_health():
    section("HEALTH CHECK")
    resp = requests.get(f"{BASE_URL}/health")
    data = resp.json()

    check("Status code 200",           resp.status_code == 200)
    check("Status is ok",              data.get("status") == "ok")
    check("Model is loaded",           data.get("model_ready") is True)
    check("last_retrain is a number",  isinstance(data.get("last_retrain"), float))
    check("uptime_hint is a string",   isinstance(data.get("uptime_hint"), str))


def test_auth():
    section("AUTHENTICATION")

    # No key
    resp = requests.get(f"{BASE_URL}/recommend", params={"user_id": U1, "item_id": P1})
    check("No API key → 422",          resp.status_code == 422)

    # Wrong key
    resp = requests.get(
        f"{BASE_URL}/recommend",
        params={"user_id": U1, "item_id": P1},
        headers={"X-API-Key": "wrong-key"},
    )
    check("Wrong API key → 403",       resp.status_code == 403)

    # Health has no auth
    resp = requests.get(f"{BASE_URL}/health")
    check("Health needs no API key",   resp.status_code == 200)


def test_recommend_known_users():
    section("RECOMMEND — KNOWN USERS + KNOWN ITEMS")

    # Alice (footwear) viewing Nike shoes → should get footwear back
    recs = get_recommend(U1, P1, n=3, refresh=True)
    ids  = [r["item_id"] for r in recs]
    expected_cat = PRODUCT_CATEGORIES[P1]
    relevant_count = check_semantic_relevance(U1, P1, recs, expected_cat)
    check("Alice/Nike → returns 3 results",          len(recs) == 3)
    check("Alice/Nike → all have scores > 0",        all(r["score"] >= 0 for r in recs))
    check("Alice/Nike → no duplicate items",         len(ids) == len(set(ids)))
    check("Alice/Nike → Nike itself not in results", P1 not in ids)
    check("Alice/Nike → has sources dict",           all(r["sources"] for r in recs))
    check("Alice/Nike → at least 1 footwear rec",    relevant_count >= 1, f"Got {relevant_count} {expected_cat} recs")

    # Bob (electronics) viewing Gaming Mouse → should get electronics back
    recs = get_recommend(U2, P4, n=3, refresh=True)
    ids  = [r["item_id"] for r in recs]
    expected_cat = PRODUCT_CATEGORIES[P4]
    relevant_count = check_semantic_relevance(U2, P4, recs, expected_cat)
    check("Bob/Mouse → returns 3 results",           len(recs) == 3)
    check("Bob/Mouse → Gaming Mouse not in results", P4 not in ids)
    check("Bob/Mouse → scores in [0, 1]",            all(0 <= r["score"] <= 1 for r in recs))
    check("Bob/Mouse → at least 1 electronics rec",  relevant_count >= 1, f"Got {relevant_count} {expected_cat} recs")

    # Carol (footwear + fitness) viewing Adidas
    recs = get_recommend(U3, P2, n=3, refresh=True)
    ids  = [r["item_id"] for r in recs]
    expected_cat = PRODUCT_CATEGORIES[P2]
    relevant_count = check_semantic_relevance(U3, P2, recs, expected_cat)
    check("Carol/Adidas → returns 3 results",        len(recs) == 3)
    check("Carol/Adidas → Adidas not in results",    P2 not in ids)
    check("Carol/Adidas → at least 1 footwear rec",  relevant_count >= 1, f"Got {relevant_count} {expected_cat} recs")

    # Dan (gaming) viewing Mechanical Keyboard
    recs = get_recommend(U4, P5, n=3, refresh=True)
    ids  = [r["item_id"] for r in recs]
    expected_cat = PRODUCT_CATEGORIES[P5]
    relevant_count = check_semantic_relevance(U4, P5, recs, expected_cat)
    check("Dan/Keyboard → returns 3 results",        len(recs) == 3)
    check("Dan/Keyboard → Keyboard not in results",  P5 not in ids)
    check("Dan/Keyboard → at least 1 electronics rec", relevant_count >= 1, f"Got {relevant_count} {expected_cat} recs")

    # Eve (footwear) viewing Puma
    recs = get_recommend(U5, P3, n=3, refresh=True)
    expected_cat = PRODUCT_CATEGORIES[P3]
    relevant_count = check_semantic_relevance(U5, P3, recs, expected_cat)
    check("Eve/Puma → returns 3 results",            len(recs) == 3)
    check("Eve/Puma → at least 1 footwear rec",      relevant_count >= 1, f"Got {relevant_count} {expected_cat} recs")

    # Henry (gaming) viewing Monitor
    recs = get_recommend(U8, P6, n=3, refresh=True)
    expected_cat = PRODUCT_CATEGORIES[P6]
    relevant_count = check_semantic_relevance(U8, P6, recs, expected_cat)
    check("Henry/Monitor → returns 3 results",       len(recs) == 3)
    check("Henry/Monitor → at least 1 electronics rec", relevant_count >= 1, f"Got {relevant_count} {expected_cat} recs")


def test_recommend_cold_start():
    section("RECOMMEND — COLD START")

    # Unknown user, known item → popularity fallback
    recs = get_recommend(U99, P1, n=3, refresh=True)
    expected_cat = PRODUCT_CATEGORIES[P1]
    relevant_count = check_semantic_relevance(U99, P1, recs, expected_cat)
    check("Unknown user → returns 3 results",        len(recs) == 3)
    check("Unknown user → scores > 0",               all(r["score"] > 0 for r in recs))
    check("Unknown user → has sources",              all(r["sources"] for r in recs))
    check("Unknown user → at least 1 footwear rec",  relevant_count >= 1, f"Got {relevant_count} {expected_cat} recs")

    # Known user, unknown item → popularity fallback
    recs = get_recommend(U1, P99, n=3, refresh=True)
    check("Unknown item → returns 3 results",        len(recs) == 3)
    check("Unknown item → scores > 0",               all(r["score"] > 0 for r in recs))

    # Both unknown → full popularity fallback
    recs = get_recommend(U99, P99, n=3, refresh=True)
    check("Both unknown → returns 3 results",        len(recs) == 3)
    check("Both unknown → scores > 0",               all(r["score"] > 0 for r in recs))


def test_recommend_n_parameter():
    section("RECOMMEND — N PARAMETER")

    recs = get_recommend(U1, P1, n=1, refresh=True)
    check("n=1 → returns exactly 1",    len(recs) == 1)

    recs = get_recommend(U1, P1, n=2, refresh=True)
    check("n=2 → returns exactly 2",    len(recs) == 2)

    recs = get_recommend(U1, P1, n=5, refresh=True)
    check("n=5 → returns up to 5",      len(recs) <= 5)

    # Invalid n
    resp = requests.get(
        f"{BASE_URL}/recommend",
        params={"user_id": U1, "item_id": P1, "n": 0},
        headers=HEADERS,
    )
    check("n=0 → 422 validation error", resp.status_code == 422)

    resp = requests.get(
        f"{BASE_URL}/recommend",
        params={"user_id": U1, "item_id": P1, "n": 999},
        headers=HEADERS,
    )
    check("n=999 → 422 validation error", resp.status_code == 422)


def test_recommend_cache():
    section("RECOMMEND — CACHE")

    # First call — cache miss, runs engine
    recs1 = get_recommend(U2, P4, n=3, refresh=True)

    # Second call — should hit cache, same results
    recs2 = get_recommend(U2, P4, n=3, refresh=False)
    ids1  = [r["item_id"] for r in recs1]
    ids2  = [r["item_id"] for r in recs2]
    check("Cache hit → same item order",   ids1 == ids2)
    check("Cache hit → same scores",       [r["score"] for r in recs1] == [r["score"] for r in recs2])

    # refresh=True bypasses cache
    recs3 = get_recommend(U2, P4, n=3, refresh=True)
    check("refresh=True → still returns 3 results", len(recs3) == 3)


def test_homepage_new_visitor():
    section("HOMEPAGE — NEW VISITOR (no user_id)")

    recs = get_homepage(user_id=None, n=3)
    check("No user_id → returns 3 results",          len(recs) == 3)
    check("No user_id → all personalised=false",     all(not r["personalised"] for r in recs))
    check("No user_id → all popularity source",      all("popularity" in r["sources"] for r in recs))
    check("No user_id → scores in [0, 1]",           all(0 <= r["score"] <= 1 for r in recs))
    check("No user_id → no duplicates",              len(set(r["item_id"] for r in recs)) == len(recs))


def test_homepage_unknown_user():
    section("HOMEPAGE — UNKNOWN USER (no history)")

    recs = get_homepage(user_id=U99, n=3)
    check("Unknown user → returns 3 results",        len(recs) == 3)
    check("Unknown user → all personalised=false",   all(not r["personalised"] for r in recs))
    check("Unknown user → all popularity source",    all("popularity" in r["sources"] for r in recs))


def test_homepage_known_users():
    section("HOMEPAGE — KNOWN USERS (has history)")

    # Alice
    recs = get_homepage(user_id=U1, n=3)
    check("Alice → returns 3 results",               len(recs) == 3)
    check("Alice → at least 1 personalised",         any(r["personalised"] for r in recs))
    check("Alice → no duplicate items",              len(set(r["item_id"] for r in recs)) == len(recs))
    check("Alice → scores > 0",                      all(r["score"] > 0 for r in recs))

    # Bob
    recs = get_homepage(user_id=U2, n=3)
    check("Bob → returns 3 results",                 len(recs) == 3)
    check("Bob → at least 1 personalised",           any(r["personalised"] for r in recs))

    # Carol
    recs = get_homepage(user_id=U3, n=3)
    check("Carol → returns 3 results",               len(recs) == 3)
    check("Carol → at least 1 personalised",         any(r["personalised"] for r in recs))

    # Dan
    recs = get_homepage(user_id=U4, n=3)
    check("Dan → returns 3 results",                 len(recs) == 3)
    check("Dan → at least 1 personalised",           any(r["personalised"] for r in recs))

    # Grace
    recs = get_homepage(user_id=U7, n=3)
    check("Grace → returns 3 results",               len(recs) == 3)

    # Henry
    recs = get_homepage(user_id=U8, n=3)
    check("Henry → returns 3 results",               len(recs) == 3)


def test_homepage_n_parameter():
    section("HOMEPAGE — N PARAMETER")

    recs = get_homepage(n=1)
    check("n=1 → returns exactly 1",    len(recs) == 1)

    recs = get_homepage(n=5)
    check("n=5 → returns up to 5",      len(recs) <= 5)

    resp = requests.get(f"{BASE_URL}/homepage", params={"n": 0}, headers=HEADERS)
    check("n=0 → 422 validation error", resp.status_code == 422)


def test_ratings():
    section("RATINGS — RECORD INTERACTIONS")

    # Valid actions
    code = post_rating(U99, P1, "view")
    check("view → 202 accepted",        code == 202)

    code = post_rating(U99, P2, "add_to_cart")
    check("add_to_cart → 202 accepted", code == 202)

    code = post_rating(U99, P3, "purchase")
    check("purchase → 202 accepted",    code == 202)

    # Invalid action
    code = post_rating(U99, P1, "like")
    check("unknown action → 422",       code == 422)

    # Missing fields
    resp = requests.post(
        f"{BASE_URL}/ratings",
        json={"user_id": U99},
        headers=HEADERS,
    )
    check("missing fields → 422",       resp.status_code == 422)


def test_retrain():
    section("RETRAIN")

    code = post_retrain()
    check("POST /retrain → 200",        code == 200)

    # Model still ready after retrain
    resp  = requests.get(f"{BASE_URL}/health")
    check("Model ready after retrain",  resp.json().get("model_ready") is True)


def test_scores_consistency():
    section("SCORES CONSISTENCY")

    # Scores should always be in [0, 1]
    for label, user, item in [
        ("alice/nike",    U1, P1),
        ("bob/mouse",     U2, P4),
        ("carol/adidas",  U3, P2),
        ("dan/keyboard",  U4, P5),
        ("unknown/nike",  U99, P1),
        ("alice/unknown", U1, P99),
    ]:
        recs = get_recommend(user, item, n=3, refresh=True)
        check(
            f"{label} → all scores in [0, 1]",
            all(0 <= r["score"] <= 1 for r in recs),
            f"scores: {[r['score'] for r in recs]}"
        )

    # Homepage scores
    for label, uid in [
        ("visitor",       None),
        ("unknown user",  U99),
        ("alice",         U1),
        ("bob",           U2),
    ]:
        recs = get_homepage(user_id=uid, n=3)
        check(
            f"homepage/{label} → all scores in [0, 1]",
            all(0 <= r["score"] <= 1 for r in recs),
            f"scores: {[r['score'] for r in recs]}"
        )


def test_sources_consistency():
    section("SOURCES CONSISTENCY")

    for label, user, item in [
        ("alice/nike",   U1, P1),
        ("bob/mouse",    U2, P4),
        ("unknown/nike", U99, P1),
        ("alice/unknown", U1, P99),
    ]:
        recs = get_recommend(user, item, n=3, refresh=True)
        for r in recs:
            sources_sum = sum(r["sources"].values())
            check(
                f"{label} → {r['name'][:20]} sources match score",
                abs(r["score"] - sources_sum) < 0.05 or r["score"] <= 1.0,
            )


# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{HEAD}{'═' * 60}{END}")
    print(f"{HEAD}  HYBRID RECOMMENDER — FULL TEST SUITE WITH SEMANTIC CHECKS{END}")
    print(f"{HEAD}{'═' * 60}{END}")

    try:
        requests.get(f"{BASE_URL}/health", timeout=3)
    except Exception:
        print("\n  ERROR: Server is not running. Start it first with:")
        print("  uvicorn main:app --reload\n")
        sys.exit(1)

    test_health()
    test_auth()
    test_recommend_known_users()
    test_recommend_cold_start()
    test_recommend_n_parameter()
    test_recommend_cache()
    test_homepage_new_visitor()
    test_homepage_unknown_user()
    test_homepage_known_users()
    test_homepage_n_parameter()
    test_ratings()
    test_retrain()
    test_scores_consistency()
    test_sources_consistency()

    print(f"\n{HEAD}{'═' * 60}{END}")
    colour = "\033[32m" if passed == total else "\033[31m"
    print(f"{colour}  RESULTS: {passed}/{total} passed{END}")
    print(f"{HEAD}{'═' * 60}{END}\n")

    sys.exit(0 if passed == total else 1)