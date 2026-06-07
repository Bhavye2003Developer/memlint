from datetime import datetime, timedelta
from stale_detector.core import StaleDetector
from stale_detector.models import FactCategory, MemoryFact, StalenessResult

NOW = datetime(2026, 6, 7)


def _fact(id: str, content: str, age_days: int,
          category: FactCategory = None, confirmation_count: int = 0) -> MemoryFact:
    return MemoryFact(
        id=id, content=content,
        created_at=NOW - timedelta(days=age_days),
        confirmation_count=confirmation_count,
        category=category,
    )


def test_check_returns_detection_report():
    detector = StaleDetector()
    facts = [
        _fact("f1", "User lives in Delhi", age_days=5),
        _fact("f2", "User works at PwC", age_days=300),
        _fact("f3", "User debugged a bug today", age_days=35),
    ]
    report = detector.check(facts, now=NOW)
    assert report.total_facts == 3
    assert report.fresh_count + report.aging_count + report.stale_count + report.expired_count == 3
    assert len(report.results) == 3


def test_check_counts_match_levels():
    detector = StaleDetector()
    # IDENTITY 5 days: 5*0.0005=0.0025 → FRESH
    # EPISODIC 30 days: 30*0.05=1.5 clamped → EXPIRED
    facts = [
        _fact("f1", "User name is Alice", age_days=5, category=FactCategory.IDENTITY),
        _fact("f2", "User debugged a bug today", age_days=30, category=FactCategory.EPISODIC),
    ]
    report = detector.check(facts, now=NOW)
    assert report.fresh_count == 1
    assert report.expired_count == 1
    assert report.aging_count == 0
    assert report.stale_count == 0


def test_filter_safe_excludes_stale_and_expired():
    detector = StaleDetector()
    facts = [
        _fact("fresh", "User name is Alice", age_days=5, category=FactCategory.IDENTITY),
        _fact("expired", "User debugged a bug today", age_days=30, category=FactCategory.EPISODIC),
    ]
    safe = detector.filter_safe(facts, now=NOW)
    safe_ids = [f.id for f in safe]
    assert "fresh" in safe_ids
    assert "expired" not in safe_ids


def test_check_one_returns_staleness_result():
    detector = StaleDetector()
    fact = _fact("f1", "User lives in Delhi", age_days=5, category=FactCategory.LOCATION)
    result = detector.check_one(fact, now=NOW)
    assert isinstance(result, StalenessResult)
    assert result.fact_id == "f1"
    assert 0.0 <= result.staleness_score <= 1.0


def test_check_one_with_context_detects_contradiction():
    detector = StaleDetector()
    old = _fact("old", "User lives in Delhi", age_days=400, category=FactCategory.LOCATION)
    new = _fact("new", "User lives in Mumbai", age_days=50, category=FactCategory.LOCATION)
    result = detector.check_one(old, context_facts=[old, new], now=NOW)
    assert result.has_contradiction is True
    assert result.contradicted_by == "new"


def test_multiple_instantiation_no_side_effects():
    d1 = StaleDetector()
    d2 = StaleDetector(use_llm=False)
    fact = _fact("f1", "User name is Alice", age_days=5, category=FactCategory.IDENTITY)
    r1 = d1.check_one(fact, now=NOW)
    r2 = d2.check_one(fact, now=NOW)
    assert r1.staleness_score == r2.staleness_score
