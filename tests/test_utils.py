import warnings
from datetime import datetime
import pytest
from memlint.utils import create_memory_metadata
from memlint.models import MemoryFact
from memlint.core import StaleDetector


def test_create_memory_metadata_with_date():
    dt = datetime(2024, 6, 1, 12, 0, 0)
    meta = create_memory_metadata(created_at=dt)
    assert meta["created_at"] == "2024-06-01T12:00:00"
    assert meta["source"] == "user"
    assert meta["confirmation_count"] == 0


def test_create_memory_metadata_custom_fields():
    dt = datetime(2024, 1, 1)
    meta = create_memory_metadata(created_at=dt, source="agent_inferred", confirmation_count=3)
    assert meta["source"] == "agent_inferred"
    assert meta["confirmation_count"] == 3


def test_create_memory_metadata_no_date_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        meta = create_memory_metadata()
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "created_at" in str(user_warnings[0].message).lower()
    assert "created_at" in meta


def test_create_memory_metadata_no_date_uses_now():
    before = datetime.utcnow()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        meta = create_memory_metadata()
    after = datetime.utcnow()
    parsed = datetime.fromisoformat(meta["created_at"])
    assert before <= parsed <= after


def test_memory_fact_created_at_optional():
    fact = MemoryFact(id="f1", content="User lives in Delhi")
    assert isinstance(fact.created_at, datetime)


def test_memory_fact_metadata_roundtrip():
    dt = datetime(2024, 3, 15)
    meta = create_memory_metadata(created_at=dt, source="agent_inferred", confirmation_count=2)
    fact = MemoryFact(id="f1", content="User works at Acme", **meta)
    assert fact.source == "agent_inferred"
    assert fact.confirmation_count == 2
    assert fact.created_at == dt


def test_detector_works_without_created_at():
    fact = MemoryFact(id="f1", content="User lives in Delhi")
    detector = StaleDetector()
    result = detector.check_one(fact)
    assert result.staleness_score >= 0.0
