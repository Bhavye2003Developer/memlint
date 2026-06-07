import json
import os
import pytest
from memlint.adapters.json_adapter import load_from_json
from memlint.adapters.mem0_adapter import load_from_mem0
from memlint.models import MemoryFact

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def test_load_from_json_returns_memory_facts():
    facts = load_from_json(os.path.join(FIXTURES, "sample.json"))
    assert len(facts) == 2
    assert all(isinstance(f, MemoryFact) for f in facts)


def test_load_from_json_parses_fields():
    facts = load_from_json(os.path.join(FIXTURES, "sample.json"))
    f = facts[0]
    assert f.id == "mem_001"
    assert f.content == "User lives in Delhi"
    assert f.confirmation_count == 0
    assert f.last_confirmed_at is None


def test_load_from_json_handles_missing_optional_fields():
    facts = load_from_json(os.path.join(FIXTURES, "sample.json"))
    f = facts[1]  # no last_confirmed_at key
    assert f.last_confirmed_at is None
    assert f.confirmation_count == 2


def test_load_from_json_raises_on_missing_id(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"content": "no id", "created_at": "2024-01-01T00:00:00"}]))
    with pytest.raises(ValueError, match="id"):
        load_from_json(str(bad))


def test_load_from_json_raises_on_missing_content(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"id": "x", "created_at": "2024-01-01T00:00:00"}]))
    with pytest.raises(ValueError, match="content"):
        load_from_json(str(bad))


def test_load_from_mem0_maps_fields():
    facts = load_from_mem0(os.path.join(FIXTURES, "sample_mem0.json"))
    assert len(facts) == 1
    f = facts[0]
    assert f.id == "abc123"
    assert f.content == "User works at PwC"
    assert f.confirmation_count == 0
    assert f.source == "user"


def test_load_from_mem0_maps_updated_at_to_last_confirmed():
    facts = load_from_mem0(os.path.join(FIXTURES, "sample_mem0.json"))
    f = facts[0]
    assert f.last_confirmed_at is not None
    assert f.last_confirmed_at.year == 2024
    assert f.last_confirmed_at.month == 10


def test_langchain_tool_importable_without_langchain():
    from memlint.adapters import langchain_tool
    assert hasattr(langchain_tool, "LANGCHAIN_AVAILABLE")
