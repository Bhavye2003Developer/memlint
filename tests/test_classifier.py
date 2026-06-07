import asyncio
from datetime import datetime
from memlint.classifier import classify_fact, classify_fact_async
from memlint.core import StaleDetector
from memlint.models import FactCategory, MemoryFact


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

    async def ainvoke(self, messages):
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


def test_memlint_accepts_llm_instance():
    llm = _MockLLM("project")
    detector = StaleDetector(use_llm=True, llm=llm)
    fact = MemoryFact(id="f1", content="anything at all", created_at=datetime(2024, 1, 1))
    result = detector.check_one(fact)
    assert result.category == FactCategory.PROJECT


def test_classify_fact_async_uses_ainvoke():
    llm = _MockLLM("employment")
    result = asyncio.run(classify_fact_async("anything at all", use_llm=True, llm=llm))
    assert result == FactCategory.EMPLOYMENT


def test_classify_fact_async_falls_back_on_bad_response():
    llm = _MockLLM("not_a_real_category")
    result = asyncio.run(classify_fact_async("User lives in Delhi", use_llm=True, llm=llm))
    assert result == FactCategory.LOCATION


def test_memlint_check_one_async():
    llm = _MockLLM("project")
    detector = StaleDetector(use_llm=True, llm=llm)
    fact = MemoryFact(id="f1", content="anything at all", created_at=datetime(2024, 1, 1))
    result = asyncio.run(detector.check_one_async(fact))
    assert result.category == FactCategory.PROJECT


def test_memlint_check_async():
    llm = _MockLLM("location")
    detector = StaleDetector(use_llm=True, llm=llm)
    facts = [
        MemoryFact(id="f1", content="a", created_at=datetime(2024, 1, 1)),
        MemoryFact(id="f2", content="b", created_at=datetime(2024, 1, 1)),
    ]
    report = asyncio.run(detector.check_async(facts))
    assert report.total_facts == 2
    assert all(r.category == FactCategory.LOCATION for r in report.results)


def test_memlint_filter_safe_async():
    llm = _MockLLM("identity")
    detector = StaleDetector(use_llm=True, llm=llm)
    facts = [MemoryFact(id="f1", content="a", created_at=datetime(2025, 6, 1))]
    safe = asyncio.run(detector.filter_safe_async(facts))
    assert isinstance(safe, list)
