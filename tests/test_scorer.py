from datetime import datetime, timedelta
from memlint.models import FactCategory, MemoryFact, StalenessLevel
from memlint.scorer import (
    compute_staleness_score, determine_level, build_reason, build_recommendation,
)

NOW = datetime(2026, 6, 7)


def _fact(id: str, content: str, age_days: int, category: FactCategory,
          confirmation_count: int = 0, source: str = "user") -> MemoryFact:
    return MemoryFact(
        id=id, content=content,
        created_at=NOW - timedelta(days=age_days),
        confirmation_count=confirmation_count,
        source=source,
        category=category,
    )


def test_fresh_identity_10_days():
    fact = _fact("f1", "User name is Alice", age_days=10, category=FactCategory.IDENTITY)
    score, _, _ = compute_staleness_score(fact, FactCategory.IDENTITY, [fact], NOW)
    assert score < 0.30


def test_expired_episodic_30_days():
    fact = _fact("f1", "User debugged a LangGraph issue today", age_days=30,
                 category=FactCategory.EPISODIC)
    score, _, _ = compute_staleness_score(fact, FactCategory.EPISODIC, [fact], NOW)
    assert score > 0.80


def test_confirmation_reduces_score():
    fact_zero = _fact("f1", "User prefers Python", age_days=200,
                      category=FactCategory.PREFERENCE, confirmation_count=0)
    fact_five = _fact("f2", "User prefers Python", age_days=200,
                      category=FactCategory.PREFERENCE, confirmation_count=5)
    score_zero, _, _ = compute_staleness_score(fact_zero, FactCategory.PREFERENCE, [fact_zero], NOW)
    score_five, _, _ = compute_staleness_score(fact_five, FactCategory.PREFERENCE, [fact_five], NOW)
    assert score_five < score_zero


def test_contradiction_detected():
    old_fact = _fact("f1", "User lives in Delhi", age_days=400, category=FactCategory.LOCATION)
    new_fact = _fact("f2", "User lives in Mumbai", age_days=50, category=FactCategory.LOCATION)
    score, has_contradiction, contradicted_by = compute_staleness_score(
        old_fact, FactCategory.LOCATION, [old_fact, new_fact], NOW
    )
    assert has_contradiction is True
    assert contradicted_by == "f2"


def test_agent_inferred_decays_faster():
    user_fact  = _fact("f1", "User works at PwC", age_days=200,
                       category=FactCategory.EMPLOYMENT, source="user")
    agent_fact = _fact("f2", "User works at PwC", age_days=200,
                       category=FactCategory.EMPLOYMENT, source="agent_inferred")
    score_user,  _, _ = compute_staleness_score(user_fact,  FactCategory.EMPLOYMENT, [user_fact],  NOW)
    score_agent, _, _ = compute_staleness_score(agent_fact, FactCategory.EMPLOYMENT, [agent_fact], NOW)
    assert score_agent > score_user


def test_determine_level_thresholds():
    assert determine_level(0.00)  == StalenessLevel.FRESH
    assert determine_level(0.29)  == StalenessLevel.FRESH
    assert determine_level(0.30)  == StalenessLevel.AGING
    assert determine_level(0.59)  == StalenessLevel.AGING
    assert determine_level(0.60)  == StalenessLevel.STALE
    assert determine_level(0.79)  == StalenessLevel.STALE
    assert determine_level(0.80)  == StalenessLevel.EXPIRED
    assert determine_level(1.00)  == StalenessLevel.EXPIRED


def test_build_recommendation():
    assert build_recommendation(StalenessLevel.FRESH)   == "use"
    assert build_recommendation(StalenessLevel.AGING)   == "verify"
    assert build_recommendation(StalenessLevel.STALE)   == "flag"
    assert build_recommendation(StalenessLevel.EXPIRED) == "discard"


def test_score_is_deterministic():
    fact = _fact("f1", "User lives in Delhi", age_days=200, category=FactCategory.LOCATION)
    r1, _, _ = compute_staleness_score(fact, FactCategory.LOCATION, [fact], NOW)
    r2, _, _ = compute_staleness_score(fact, FactCategory.LOCATION, [fact], NOW)
    assert r1 == r2


def test_score_clamped_to_one():
    # EPISODIC 365 days: 365 * 0.05 = 18.25 — must clamp to 1.0
    fact = _fact("f1", "User fixed a bug today", age_days=365, category=FactCategory.EPISODIC)
    score, _, _ = compute_staleness_score(fact, FactCategory.EPISODIC, [fact], NOW)
    assert score == 1.0
