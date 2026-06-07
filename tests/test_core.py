from datetime import datetime, timedelta
from memlint.core import StaleDetector
from memlint.models import FactCategory, MemoryFact, StalenessResult

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
        _fact("f2", "User works at Acme Corp", age_days=300),
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


# configurable decay rates

def test_custom_decay_rate_lowers_score():
    fact = _fact("f1", "User works at Acme", age_days=100, category=FactCategory.EMPLOYMENT)
    default_score = StaleDetector().check_one(fact, now=NOW).staleness_score
    slow_score = StaleDetector(decay_rates={FactCategory.EMPLOYMENT: 0.001}).check_one(fact, now=NOW).staleness_score
    assert slow_score < default_score


def test_custom_decay_rate_raises_score():
    fact = _fact("f1", "User works at Acme", age_days=100, category=FactCategory.EMPLOYMENT)
    default_score = StaleDetector().check_one(fact, now=NOW).staleness_score
    fast_score = StaleDetector(decay_rates={FactCategory.EMPLOYMENT: 0.010}).check_one(fact, now=NOW).staleness_score
    assert fast_score > default_score


def test_custom_decay_rate_only_affects_specified_category():
    location_fact = _fact("f1", "User lives in Delhi", age_days=100, category=FactCategory.LOCATION)
    default_score = StaleDetector().check_one(location_fact, now=NOW).staleness_score
    # override employment only, location should be unchanged
    custom_score = StaleDetector(decay_rates={FactCategory.EMPLOYMENT: 0.001}).check_one(location_fact, now=NOW).staleness_score
    assert custom_score == default_score


def test_custom_decay_rate_no_global_mutation():
    StaleDetector(decay_rates={FactCategory.EMPLOYMENT: 0.001})
    fact = _fact("f1", "User works at Acme", age_days=100, category=FactCategory.EMPLOYMENT)
    score_after = StaleDetector().check_one(fact, now=NOW).staleness_score
    # default detector must still use original rate
    from memlint.scorer import DECAY_RATES
    expected = min(100 * DECAY_RATES[FactCategory.EMPLOYMENT], 1.0)
    assert abs(score_after - expected) < 0.001


# when_stale

def test_when_stale_returns_three_keys():
    fact = _fact("f1", "User works at Acme", age_days=0, category=FactCategory.EMPLOYMENT)
    schedule = StaleDetector().when_stale(fact, now=NOW)
    assert set(schedule.keys()) == {"aging", "stale", "expired"}


def test_when_stale_dates_are_ordered():
    fact = _fact("f1", "User works at Acme", age_days=0, category=FactCategory.EMPLOYMENT)
    s = StaleDetector().when_stale(fact, now=NOW)
    assert s["aging"] < s["stale"] < s["expired"]


def test_when_stale_fresh_fact_all_future():
    fact = _fact("f1", "User works at Acme", age_days=0, category=FactCategory.EMPLOYMENT)
    s = StaleDetector().when_stale(fact, now=NOW)
    assert s["aging"] > NOW
    assert s["stale"] > NOW
    assert s["expired"] > NOW


def test_when_stale_already_expired_fact():
    fact = _fact("f1", "User debugged a bug", age_days=60, category=FactCategory.EPISODIC)
    s = StaleDetector().when_stale(fact, now=NOW)
    assert s["expired"] < NOW


def test_when_stale_respects_custom_decay_rates():
    fact = _fact("f1", "User works at Acme", age_days=0, category=FactCategory.EMPLOYMENT)
    s_default = StaleDetector().when_stale(fact, now=NOW)
    s_slow = StaleDetector(decay_rates={FactCategory.EMPLOYMENT: 0.001}).when_stale(fact, now=NOW)
    assert s_slow["aging"] > s_default["aging"]
