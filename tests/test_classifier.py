from datetime import datetime
from stale_detector.classifier import classify_fact
from stale_detector.core import StaleDetector
from stale_detector.models import FactCategory, MemoryFact


class _MockLLM:
    """Minimal LangChain-compatible LLM stub for testing."""
    def __init__(self, response: str):
        self._response = response

    def invoke(self, messages):
        class _Resp:
            pass
        r = _Resp()
        r.content = self._response
        return r


def test_classifies_location():
    assert classify_fact("User lives in Delhi") == FactCategory.LOCATION


def test_classifies_employment():
    assert classify_fact("User works at PwC as a consultant") == FactCategory.EMPLOYMENT


def test_classifies_project():
    assert classify_fact("Building pract-agents using LangGraph") == FactCategory.PROJECT


def test_classifies_preference():
    assert classify_fact("User prefers Python over JavaScript") == FactCategory.PREFERENCE


def test_unknown_fallback():
    assert classify_fact("The sky is blue") == FactCategory.UNKNOWN


def test_case_insensitive():
    assert classify_fact("USER LIVES IN DELHI") == FactCategory.LOCATION


def test_most_hits_wins():
    # EMPLOYMENT has: "works at", "title", "analyst" = 3 hits; PREFERENCE has: "prefers" = 1 hit
    assert classify_fact("User works at company, title is analyst, prefers Python") == FactCategory.EMPLOYMENT


def test_classify_fact_uses_provided_llm():
    llm = _MockLLM("employment")
    result = classify_fact("anything at all", use_llm=True, llm=llm)
    assert result == FactCategory.EMPLOYMENT


def test_classify_fact_llm_falls_back_on_bad_response():
    # LLM returns garbage → fall back to rule-based
    llm = _MockLLM("not_a_real_category")
    result = classify_fact("User lives in Delhi", use_llm=True, llm=llm)
    assert result == FactCategory.LOCATION


def test_stale_detector_accepts_llm_instance():
    llm = _MockLLM("project")
    detector = StaleDetector(use_llm=True, llm=llm)
    fact = MemoryFact(id="f1", content="anything at all", created_at=datetime(2024, 1, 1))
    result = detector.check_one(fact)
    assert result.category == FactCategory.PROJECT
