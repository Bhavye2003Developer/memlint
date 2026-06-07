from datetime import datetime, timedelta
from memlint.models import FactCategory, MemoryFact
from memlint.scorer import compute_staleness_score

NOW = datetime(2026, 6, 7)


def _fact(id: str, content: str, age_days: int, category: FactCategory) -> MemoryFact:
    return MemoryFact(
        id=id, content=content,
        created_at=NOW - timedelta(days=age_days),
        category=category,
    )


# negation signal + shared entity

def test_negation_left_company():
    old = _fact("f1", "User works at Acme Corp", age_days=300, category=FactCategory.EMPLOYMENT)
    new = _fact("f2", "User left Acme Corp", age_days=10, category=FactCategory.EMPLOYMENT)
    _, has_contradiction, _ = compute_staleness_score(old, FactCategory.EMPLOYMENT, [old, new], NOW)
    assert has_contradiction is True


def test_negation_moved_from_city():
    old = _fact("f1", "User lives in Delhi", age_days=300, category=FactCategory.LOCATION)
    new = _fact("f2", "User moved from Delhi to Mumbai", age_days=10, category=FactCategory.LOCATION)
    _, has_contradiction, _ = compute_staleness_score(old, FactCategory.LOCATION, [old, new], NOW)
    assert has_contradiction is True


def test_negation_no_longer():
    old = _fact("f1", "User works at Google", age_days=200, category=FactCategory.EMPLOYMENT)
    new = _fact("f2", "User no longer works at Google", age_days=5, category=FactCategory.EMPLOYMENT)
    _, has_contradiction, _ = compute_staleness_score(old, FactCategory.EMPLOYMENT, [old, new], NOW)
    assert has_contradiction is True


def test_negation_quit():
    old = _fact("f1", "User works at Meta", age_days=150, category=FactCategory.EMPLOYMENT)
    new = _fact("f2", "User quit Meta last month", age_days=5, category=FactCategory.EMPLOYMENT)
    _, has_contradiction, _ = compute_staleness_score(old, FactCategory.EMPLOYMENT, [old, new], NOW)
    assert has_contradiction is True


# transition keyword + anchor keyword

def test_transition_switched_project():
    old = _fact("f1", "User is building a project with Pinecone", age_days=200, category=FactCategory.PROJECT)
    new = _fact("f2", "User migrated from Pinecone to Qdrant", age_days=5, category=FactCategory.PROJECT)
    _, has_contradiction, _ = compute_staleness_score(old, FactCategory.PROJECT, [old, new], NOW)
    assert has_contradiction is True


def test_transition_location_relocated():
    old = _fact("f1", "User is based in Delhi", age_days=200, category=FactCategory.LOCATION)
    new = _fact("f2", "User relocated from Delhi", age_days=10, category=FactCategory.LOCATION)
    _, has_contradiction, _ = compute_staleness_score(old, FactCategory.LOCATION, [old, new], NOW)
    assert has_contradiction is True


# no contradiction when entity does not match

def test_no_contradiction_different_entity():
    old = _fact("f1", "User works at Acme Corp", age_days=200, category=FactCategory.EMPLOYMENT)
    new = _fact("f2", "User left Google", age_days=10, category=FactCategory.EMPLOYMENT)
    _, has_contradiction, _ = compute_staleness_score(old, FactCategory.EMPLOYMENT, [old, new], NOW)
    assert has_contradiction is False


# negation on newer fact penalizes older fact, not the other way round

def test_newer_negation_penalizes_older():
    old = _fact("f1", "User works at Acme Corp", age_days=300, category=FactCategory.EMPLOYMENT)
    new = _fact("f2", "User left Acme Corp", age_days=5, category=FactCategory.EMPLOYMENT)
    score_old, _, _ = compute_staleness_score(old, FactCategory.EMPLOYMENT, [old, new], NOW)
    score_new, _, _ = compute_staleness_score(new, FactCategory.EMPLOYMENT, [old, new], NOW)
    assert score_old > score_new
