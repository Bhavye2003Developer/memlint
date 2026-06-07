from datetime import datetime
from memlint.utils import confirm_fact, confirm_facts
from memlint.models import MemoryFact
from memlint.core import StaleDetector


def _fact(days_old: int, confirmations: int = 0) -> MemoryFact:
    from datetime import timedelta
    return MemoryFact(
        id="f1",
        content="User works at Acme",
        created_at=datetime.utcnow() - timedelta(days=days_old),
        confirmation_count=confirmations,
    )


# confirm_fact tests

def test_confirm_fact_increments_count():
    fact = _fact(30)
    updated = confirm_fact(fact)
    assert updated.confirmation_count == 1


def test_confirm_fact_sets_last_confirmed_at():
    fact = _fact(30)
    before = datetime.utcnow()
    updated = confirm_fact(fact)
    after = datetime.utcnow()
    assert before <= updated.last_confirmed_at <= after


def test_confirm_fact_does_not_mutate_original():
    fact = _fact(30)
    confirm_fact(fact)
    assert fact.confirmation_count == 0
    assert fact.last_confirmed_at is None


def test_confirm_fact_stacks():
    fact = _fact(30, confirmations=2)
    updated = confirm_fact(fact)
    assert updated.confirmation_count == 3


def test_confirm_fact_lowers_staleness_score():
    detector = StaleDetector()
    fact = _fact(120)
    before_score = detector.check_one(fact).staleness_score
    confirmed = confirm_fact(fact)
    after_score = detector.check_one(confirmed).staleness_score
    assert after_score < before_score


def test_confirm_facts_batch():
    facts = [_fact(30), _fact(60), _fact(90)]
    updated = confirm_facts(facts)
    assert all(f.confirmation_count == 1 for f in updated)
    assert len(updated) == 3


def test_confirm_facts_does_not_mutate_originals():
    facts = [_fact(30), _fact(60)]
    confirm_facts(facts)
    assert all(f.confirmation_count == 0 for f in facts)


# export_scores tests

def test_export_scores_returns_list():
    facts = [_fact(30), _fact(200)]
    detector = StaleDetector()
    report = detector.check(facts)
    scores = report.export_scores()
    assert len(scores) == 2


def test_export_scores_keys():
    fact = _fact(30)
    detector = StaleDetector()
    report = detector.check([fact])
    entry = report.export_scores()[0]
    assert "fact_id" in entry
    assert "memlint_score" in entry
    assert "memlint_level" in entry
    assert "memlint_age_days" in entry
    assert "memlint_checked_at" in entry


def test_export_scores_values():
    fact = _fact(30)
    detector = StaleDetector()
    report = detector.check([fact])
    entry = report.export_scores()[0]
    assert entry["fact_id"] == "f1"
    assert isinstance(entry["memlint_score"], float)
    assert entry["memlint_level"] in ("fresh", "aging", "stale", "expired")
    assert isinstance(entry["memlint_age_days"], int)
