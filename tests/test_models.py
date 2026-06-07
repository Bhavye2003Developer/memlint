from datetime import datetime
from stale_detector.models import (
    FactCategory, StalenessLevel, MemoryFact, StalenessResult, DetectionReport,
)


def test_memory_fact_defaults():
    fact = MemoryFact(id="m1", content="User lives in Delhi", created_at=datetime(2024, 1, 1))
    assert fact.confirmation_count == 0
    assert fact.source == "user"
    assert fact.category is None
    assert fact.last_confirmed_at is None
    assert fact.metadata == {}


def test_staleness_level_enum_values():
    assert StalenessLevel.FRESH == "fresh"
    assert StalenessLevel.AGING == "aging"
    assert StalenessLevel.STALE == "stale"
    assert StalenessLevel.EXPIRED == "expired"


def test_fact_category_enum_values():
    assert FactCategory.LOCATION == "location"
    assert FactCategory.EMPLOYMENT == "employment"
    assert FactCategory.UNKNOWN == "unknown"


def test_detection_report_flagged_and_safe():
    result_fresh = StalenessResult(
        fact_id="f1", content="x", category=FactCategory.IDENTITY,
        staleness_score=0.1, staleness_level=StalenessLevel.FRESH,
        age_days=10, reason="r", recommendation="use",
    )
    result_aging = StalenessResult(
        fact_id="f2", content="y", category=FactCategory.PREFERENCE,
        staleness_score=0.4, staleness_level=StalenessLevel.AGING,
        age_days=130, reason="r", recommendation="verify",
    )
    result_stale = StalenessResult(
        fact_id="f3", content="z", category=FactCategory.EMPLOYMENT,
        staleness_score=0.7, staleness_level=StalenessLevel.STALE,
        age_days=280, reason="r", recommendation="flag",
    )
    result_expired = StalenessResult(
        fact_id="f4", content="w", category=FactCategory.EPISODIC,
        staleness_score=1.0, staleness_level=StalenessLevel.EXPIRED,
        age_days=30, reason="r", recommendation="discard",
    )
    report = DetectionReport(
        checked_at=datetime(2026, 1, 1),
        total_facts=4, fresh_count=1, aging_count=1, stale_count=1, expired_count=1,
        results=[result_fresh, result_aging, result_stale, result_expired],
    )
    assert len(report.flagged) == 2
    assert {r.fact_id for r in report.flagged} == {"f3", "f4"}
    assert len(report.safe) == 2
    assert {r.fact_id for r in report.safe} == {"f1", "f2"}
