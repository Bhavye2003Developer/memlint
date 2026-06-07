# Stale Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `stale-detector`, a Python library + CLI that classifies memory facts by category, computes staleness scores, and flags outdated memories before LLM injection.

**Architecture:** Data flows through three layers: `models.py` defines Pydantic v2 schemas, `classifier.py` assigns a `FactCategory` to each fact via keyword matching or an optional LLM call, and `scorer.py` computes a 0–1 staleness score using decay rates, confirmation bonuses, source penalties, and contradiction detection. `core.py` orchestrates these layers. Adapters and a click+rich CLI sit at the edges.

**Tech Stack:** Python 3.11+, Pydantic v2, click, rich, python-dateutil, python-dotenv. Optional: langchain-core + langchain-openai for LLM classification and tool wrapping.

---

## File Map

| Path | Action | Purpose |
|------|--------|---------|
| `pyproject.toml` | Create | Project metadata, deps, CLI entry point |
| `.env.example` | Create | Template for optional API keys |
| `stale_detector/__init__.py` | Create | Public API re-exports |
| `stale_detector/models.py` | Create | Pydantic models: MemoryFact, StalenessResult, DetectionReport, enums |
| `stale_detector/classifier.py` | Create | Rule-based + optional LLM fact classification |
| `stale_detector/scorer.py` | Create | Staleness scoring: decay, confirmation, contradiction |
| `stale_detector/core.py` | Create | StaleDetector orchestrator |
| `stale_detector/adapters/__init__.py` | Create | Empty |
| `stale_detector/adapters/json_adapter.py` | Create | Load MemoryFact list from JSON |
| `stale_detector/adapters/mem0_adapter.py` | Create | Load from Mem0 JSON format |
| `stale_detector/adapters/langchain_tool.py` | Create | LangChain tool wrappers (guarded import) |
| `stale_detector/cli.py` | Create | click + rich CLI |
| `tests/__init__.py` | Create | Empty |
| `tests/test_models.py` | Create | Model construction and property tests |
| `tests/test_classifier.py` | Create | Classifier unit tests |
| `tests/test_scorer.py` | Create | Scorer unit tests |
| `tests/test_core.py` | Create | Core integration tests |
| `tests/test_adapters.py` | Create | Adapter loading tests |
| `tests/fixtures/sample.json` | Create | JSON adapter test fixture |
| `tests/fixtures/sample_mem0.json` | Create | Mem0 adapter test fixture |
| `examples/sample_memories.json` | Create | 8 sample facts with all staleness levels |
| `examples/basic_usage.py` | Create | Standalone usage example |
| `examples/langchain_integration.py` | Create | LangGraph node example |
| `README.md` | Create | Full project documentation (9 sections) |

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `stale_detector/__init__.py`
- Create: `stale_detector/adapters/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "stale-detector"
version = "0.1.0"
description = "Detect stale facts in LLM agent memory stores"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.0",
    "python-dateutil>=2.8",
    "click>=8.0",
    "rich>=13.0",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
llm = [
    "langchain-core>=0.2",
    "langchain-openai>=0.1",
]
dev = [
    "pytest>=8.0",
    "pytest-cov",
]

[project.scripts]
stale-detector = "stale_detector.cli:main"
```

- [ ] **Step 2: Create .env.example**

```
# Optional: only needed if use_llm=True
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

- [ ] **Step 3: Create empty init files**

Create `stale_detector/__init__.py` (empty for now):
```python
```

Create `stale_detector/adapters/__init__.py`:
```python
```

Create `tests/__init__.py`:
```python
```

- [ ] **Step 4: Install in dev mode**

```bash
pip install -e ".[dev]"
```

Expected: installs successfully, no errors.

- [ ] **Step 5: Verify pytest baseline**

```bash
pytest
```

Expected: `no tests ran` — 0 collected, 0 errors.

- [ ] **Step 6: Commit**

```bash
git init
git add pyproject.toml .env.example stale_detector/__init__.py stale_detector/adapters/__init__.py tests/__init__.py
git commit -m "chore: project scaffolding"
```

---

### Task 2: Data Models

**Files:**
- Create: `stale_detector/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_models.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_models.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'stale_detector.models'`

- [ ] **Step 3: Implement models.py**

Create `stale_detector/models.py`:
```python
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import Optional
from enum import Enum


class FactCategory(str, Enum):
    LOCATION = "location"
    EMPLOYMENT = "employment"
    PROJECT = "project"
    PREFERENCE = "preference"
    RELATIONSHIP = "relationship"
    IDENTITY = "identity"
    EPISODIC = "episodic"
    SYSTEM_FACT = "system_fact"
    UNKNOWN = "unknown"


class StalenessLevel(str, Enum):
    FRESH = "fresh"
    AGING = "aging"
    STALE = "stale"
    EXPIRED = "expired"


class MemoryFact(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    id: str
    content: str
    created_at: datetime
    last_confirmed_at: Optional[datetime] = None
    confirmation_count: int = 0
    category: Optional[FactCategory] = None
    source: str = "user"
    metadata: dict = Field(default_factory=dict)


class StalenessResult(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    fact_id: str
    content: str
    category: FactCategory
    staleness_score: float
    staleness_level: StalenessLevel
    age_days: int
    reason: str
    recommendation: str
    has_contradiction: bool = False
    contradicted_by: Optional[str] = None


class DetectionReport(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    checked_at: datetime
    total_facts: int
    fresh_count: int
    aging_count: int
    stale_count: int
    expired_count: int
    results: list[StalenessResult]

    @property
    def flagged(self) -> list[StalenessResult]:
        return [r for r in self.results
                if r.staleness_level in (StalenessLevel.STALE, StalenessLevel.EXPIRED)]

    @property
    def safe(self) -> list[StalenessResult]:
        return [r for r in self.results
                if r.staleness_level in (StalenessLevel.FRESH, StalenessLevel.AGING)]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_models.py -v
```

Expected: PASS — 4 tests.

- [ ] **Step 5: Commit**

```bash
git add stale_detector/models.py tests/test_models.py
git commit -m "feat: add Pydantic data models"
```

---

### Task 3: Classifier

**Files:**
- Create: `stale_detector/classifier.py`
- Create: `tests/test_classifier.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_classifier.py`:
```python
from stale_detector.classifier import classify_fact
from stale_detector.models import FactCategory


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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_classifier.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'stale_detector.classifier'`

- [ ] **Step 3: Implement classifier.py**

Create `stale_detector/classifier.py`:
```python
from stale_detector.models import FactCategory

CATEGORY_KEYWORDS: dict[FactCategory, list[str]] = {
    FactCategory.LOCATION: [
        "lives", "located", "based in", "address", "city", "country",
        "office", "moved to", "residing", "hometown", "location",
    ],
    FactCategory.EMPLOYMENT: [
        "works at", "employed", "job", "role", "position", "company",
        "organization", "joined", "hired", "manager", "team", "department",
        "title", "consultant", "engineer", "analyst", "intern",
    ],
    FactCategory.PROJECT: [
        "project", "building", "repo", "codebase", "app", "tool",
        "working on", "developing", "implementing", "stack", "framework",
        "library", "version", "api", "endpoint", "deployed", "launched",
    ],
    FactCategory.PREFERENCE: [
        "prefers", "likes", "favorite", "enjoys", "uses", "dislikes",
        "wants", "chooses", "opts for", "theme", "mode", "setting",
        "style", "approach",
    ],
    FactCategory.RELATIONSHIP: [
        "friend", "colleague", "manager", "reports to", "partner",
        "teammate", "mentor", "client", "collaborator", "family",
    ],
    FactCategory.IDENTITY: [
        "name is", "called", "age", "born", "nationality", "speaks",
        "gender", "education", "degree", "graduated", "alumni",
    ],
    FactCategory.EPISODIC: [
        "today", "yesterday", "last week", "this morning", "just",
        "recently", "earlier", "said that", "mentioned", "asked about",
        "discussed", "fixed", "resolved", "debugging",
    ],
    FactCategory.SYSTEM_FACT: [
        "python version", "node version", "npm", "pip", "docker",
        "os", "operating system", "machine", "cpu", "ram", "disk",
        "installed", "configured", "environment", "env", ".env",
    ],
}

CLASSIFY_PROMPT = """You are classifying a memory fact into exactly one category.

Categories:
- location: where someone lives, works, or is based
- employment: job, company, role, title, team
- project: software projects, tools being built, tech stack
- preference: likes, dislikes, habits, settings
- relationship: people the user knows or works with
- identity: name, age, education, nationality, languages spoken
- episodic: time-specific events, recent actions, things that happened
- system_fact: software versions, OS, environment config
- unknown: does not fit any category

Memory fact: "{fact}"

Respond with ONLY the category name, nothing else. Example: "employment"
"""


def _rule_based_classify(content: str) -> FactCategory:
    lower = content.lower()
    scores: dict[FactCategory, int] = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in lower)
        if hits > 0:
            scores[category] = hits
    if not scores:
        return FactCategory.UNKNOWN
    return max(scores, key=lambda c: scores[c])


def _llm_classify(content: str, llm_provider: str, model: str) -> FactCategory:
    import os
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    api_key = (
        os.getenv("OPENAI_API_KEY") if llm_provider == "openai"
        else os.getenv("ANTHROPIC_API_KEY")
    )
    if not api_key:
        raise ValueError(f"No API key found for provider {llm_provider!r}")

    llm = ChatOpenAI(model=model, temperature=0, api_key=api_key)
    response = llm.invoke([HumanMessage(content=CLASSIFY_PROMPT.format(fact=content))])
    raw = response.content.strip().lower()
    try:
        return FactCategory(raw)
    except ValueError:
        raise ValueError(f"LLM returned unrecognized category: {raw!r}")


def classify_fact(
    content: str,
    use_llm: bool = False,
    llm_provider: str = "openai",
    model: str = "gpt-4o-mini",
) -> FactCategory:
    if use_llm:
        try:
            return _llm_classify(content, llm_provider, model)
        except Exception:
            pass
    return _rule_based_classify(content)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_classifier.py -v
```

Expected: PASS — 7 tests.

- [ ] **Step 5: Commit**

```bash
git add stale_detector/classifier.py tests/test_classifier.py
git commit -m "feat: add fact classifier with rule-based and optional LLM modes"
```

---

### Task 4: Scorer

**Files:**
- Create: `stale_detector/scorer.py`
- Create: `tests/test_scorer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scorer.py`:
```python
from datetime import datetime, timedelta
from stale_detector.models import FactCategory, MemoryFact, StalenessLevel
from stale_detector.scorer import (
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_scorer.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'stale_detector.scorer'`

- [ ] **Step 3: Implement scorer.py**

Create `stale_detector/scorer.py`:
```python
from datetime import datetime
from stale_detector.models import FactCategory, MemoryFact, StalenessLevel
from stale_detector.classifier import CATEGORY_KEYWORDS

DECAY_RATES: dict[FactCategory, float] = {
    FactCategory.LOCATION:      0.0020,
    FactCategory.EMPLOYMENT:    0.0025,
    FactCategory.PROJECT:       0.0060,
    FactCategory.PREFERENCE:    0.0030,
    FactCategory.RELATIONSHIP:  0.0025,
    FactCategory.IDENTITY:      0.0005,
    FactCategory.EPISODIC:      0.0500,
    FactCategory.SYSTEM_FACT:   0.0100,
    FactCategory.UNKNOWN:       0.0030,
}


def determine_level(score: float) -> StalenessLevel:
    if score < 0.30:
        return StalenessLevel.FRESH
    if score < 0.60:
        return StalenessLevel.AGING
    if score < 0.80:
        return StalenessLevel.STALE
    return StalenessLevel.EXPIRED


def build_reason(
    age_days: int,
    category: FactCategory,
    confirmation_count: int,
    has_contradiction: bool,
    score: float,
) -> str:
    decay_days = int(1 / DECAY_RATES[category])
    parts = [f"{age_days} days old ({category.value} facts decay in ~{decay_days} days)"]
    if confirmation_count > 0:
        parts.append(f"confirmed {confirmation_count} time(s)")
    if has_contradiction:
        parts.append("a newer conflicting fact exists")
    return "; ".join(parts)


def build_recommendation(level: StalenessLevel) -> str:
    return {
        StalenessLevel.FRESH:   "use",
        StalenessLevel.AGING:   "verify",
        StalenessLevel.STALE:   "flag",
        StalenessLevel.EXPIRED: "discard",
    }[level]


def _are_contradictory(
    fact_a: MemoryFact,
    fact_b: MemoryFact,
    category: FactCategory,
) -> bool:
    if fact_a.category != fact_b.category and not (
        fact_a.category == category or fact_b.category == category
    ):
        return False

    time_diff = abs((fact_a.created_at - fact_b.created_at).days)
    if time_diff < 1:
        return False

    keywords = CATEGORY_KEYWORDS.get(category, [])
    a_lower = fact_a.content.lower()
    b_lower = fact_b.content.lower()
    return any(kw in a_lower and kw in b_lower for kw in keywords)


def compute_staleness_score(
    fact: MemoryFact,
    category: FactCategory,
    all_facts: list[MemoryFact],
    now: datetime,
) -> tuple[float, bool, str | None]:
    """Returns (score, has_contradiction, contradicted_by_id)."""
    reference_time = fact.last_confirmed_at or fact.created_at
    age_days = max((now - reference_time).days, 0)

    decay_rate = DECAY_RATES[category]
    score = age_days * decay_rate

    confirmation_reduction = min(fact.confirmation_count * 0.08, 0.40)
    score -= confirmation_reduction

    if fact.source == "agent_inferred":
        score *= 1.3

    has_contradiction = False
    contradicted_by: str | None = None
    for other in all_facts:
        if other.id == fact.id:
            continue
        if _are_contradictory(fact, other, category):
            has_contradiction = True
            contradicted_by = other.id
            score += 0.40
            break

    return min(max(score, 0.0), 1.0), has_contradiction, contradicted_by
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_scorer.py -v
```

Expected: PASS — 10 tests.

- [ ] **Step 5: Commit**

```bash
git add stale_detector/scorer.py tests/test_scorer.py
git commit -m "feat: add staleness scorer with decay, confirmations, and contradiction detection"
```

---

### Task 5: Core Class

**Files:**
- Create: `stale_detector/core.py`
- Create: `tests/test_core.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_core.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_core.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'stale_detector.core'`

- [ ] **Step 3: Implement core.py**

Create `stale_detector/core.py`:
```python
from datetime import datetime
from stale_detector.models import (
    FactCategory, MemoryFact, StalenessResult, DetectionReport, StalenessLevel,
)
from stale_detector.classifier import classify_fact
from stale_detector.scorer import (
    compute_staleness_score, determine_level, build_reason, build_recommendation,
)


class StaleDetector:
    def __init__(
        self,
        use_llm: bool = False,
        llm_provider: str = "openai",
        model: str = "gpt-4o-mini",
    ):
        self._use_llm = use_llm
        self._llm_provider = llm_provider
        self._model = model

    def _classify(self, fact: MemoryFact) -> FactCategory:
        if fact.category is not None:
            return fact.category
        return classify_fact(
            fact.content,
            use_llm=self._use_llm,
            llm_provider=self._llm_provider,
            model=self._model,
        )

    def check_one(
        self,
        fact: MemoryFact,
        context_facts: list[MemoryFact] | None = None,
        now: datetime | None = None,
    ) -> StalenessResult:
        if now is None:
            now = datetime.utcnow()
        all_facts = context_facts if context_facts is not None else [fact]
        category = self._classify(fact)
        score, has_contradiction, contradicted_by = compute_staleness_score(
            fact, category, all_facts, now
        )
        level = determine_level(score)
        reference_time = fact.last_confirmed_at or fact.created_at
        age_days = max((now - reference_time).days, 0)
        return StalenessResult(
            fact_id=fact.id,
            content=fact.content,
            category=category,
            staleness_score=round(score, 4),
            staleness_level=level,
            age_days=age_days,
            reason=build_reason(age_days, category, fact.confirmation_count, has_contradiction, score),
            recommendation=build_recommendation(level),
            has_contradiction=has_contradiction,
            contradicted_by=contradicted_by,
        )

    def check(
        self,
        facts: list[MemoryFact],
        now: datetime | None = None,
    ) -> DetectionReport:
        if now is None:
            now = datetime.utcnow()
        results = [self.check_one(f, context_facts=facts, now=now) for f in facts]
        counts: dict[StalenessLevel, int] = {level: 0 for level in StalenessLevel}
        for r in results:
            counts[r.staleness_level] += 1
        return DetectionReport(
            checked_at=now,
            total_facts=len(facts),
            fresh_count=counts[StalenessLevel.FRESH],
            aging_count=counts[StalenessLevel.AGING],
            stale_count=counts[StalenessLevel.STALE],
            expired_count=counts[StalenessLevel.EXPIRED],
            results=results,
        )

    def filter_safe(
        self,
        facts: list[MemoryFact],
        now: datetime | None = None,
    ) -> list[MemoryFact]:
        report = self.check(facts, now)
        safe_ids = {r.fact_id for r in report.safe}
        return [f for f in facts if f.id in safe_ids]
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_core.py -v
```

Expected: PASS — 6 tests.

- [ ] **Step 5: Run full suite**

```bash
pytest -v
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add stale_detector/core.py tests/test_core.py
git commit -m "feat: add StaleDetector core class"
```

---

### Task 6: JSON and Mem0 Adapters

**Files:**
- Create: `stale_detector/adapters/json_adapter.py`
- Create: `stale_detector/adapters/mem0_adapter.py`
- Create: `tests/test_adapters.py`
- Create: `tests/fixtures/sample.json`
- Create: `tests/fixtures/sample_mem0.json`

- [ ] **Step 1: Create test fixtures**

Create `tests/fixtures/sample.json`:
```json
[
  {
    "id": "mem_001",
    "content": "User lives in Delhi",
    "created_at": "2024-08-15T10:00:00",
    "last_confirmed_at": null,
    "confirmation_count": 0,
    "source": "user"
  },
  {
    "id": "mem_002",
    "content": "User prefers Python over JavaScript",
    "created_at": "2025-01-10T08:30:00",
    "confirmation_count": 2,
    "source": "user"
  }
]
```

Create `tests/fixtures/sample_mem0.json`:
```json
[
  {
    "id": "abc123",
    "memory": "User works at PwC",
    "created_at": "2024-09-01T00:00:00",
    "updated_at": "2024-10-15T00:00:00",
    "metadata": {}
  }
]
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_adapters.py`:
```python
import json
import os
import pytest
from stale_detector.adapters.json_adapter import load_from_json
from stale_detector.adapters.mem0_adapter import load_from_mem0
from stale_detector.models import MemoryFact

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
    from stale_detector.adapters import langchain_tool
    assert hasattr(langchain_tool, "LANGCHAIN_AVAILABLE")
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_adapters.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement json_adapter.py**

Create `stale_detector/adapters/json_adapter.py`:
```python
import json
from datetime import datetime
from dateutil import parser as dateutil_parser
from stale_detector.models import MemoryFact


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return dateutil_parser.parse(value).replace(tzinfo=None)


def load_from_json(filepath: str) -> list[MemoryFact]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    facts = []
    for i, entry in enumerate(data):
        for required in ("id", "content", "created_at"):
            if required not in entry:
                raise ValueError(f"Entry {i} missing required field '{required}'")

        facts.append(MemoryFact(
            id=entry["id"],
            content=entry["content"],
            created_at=_parse_dt(entry["created_at"]),
            last_confirmed_at=_parse_dt(entry.get("last_confirmed_at")),
            confirmation_count=entry.get("confirmation_count", 0),
            source=entry.get("source", "user"),
            metadata=entry.get("metadata", {}),
        ))
    return facts
```

- [ ] **Step 5: Implement mem0_adapter.py**

Create `stale_detector/adapters/mem0_adapter.py`:
```python
import json
from datetime import datetime
from dateutil import parser as dateutil_parser
from stale_detector.models import MemoryFact


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return dateutil_parser.parse(value).replace(tzinfo=None)


def load_from_mem0(filepath: str) -> list[MemoryFact]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    facts = []
    for i, entry in enumerate(data):
        for required in ("id", "memory", "created_at"):
            if required not in entry:
                raise ValueError(f"Entry {i} missing required field '{required}'")

        facts.append(MemoryFact(
            id=entry["id"],
            content=entry["memory"],
            created_at=_parse_dt(entry["created_at"]),
            last_confirmed_at=_parse_dt(entry.get("updated_at")),
            confirmation_count=entry.get("confirmation_count", 0),
            source=entry.get("source", "user"),
            metadata=entry.get("metadata", {}),
        ))
    return facts
```

- [ ] **Step 6: Implement langchain_tool.py**

Create `stale_detector/adapters/langchain_tool.py`:
```python
import json
from stale_detector.models import MemoryFact

try:
    from langchain_core.tools import tool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

if LANGCHAIN_AVAILABLE:
    from stale_detector.core import StaleDetector

    @tool
    def check_memory_staleness(fact_json: str) -> str:
        """Check if a single memory fact is stale before injecting it into context."""
        fact = MemoryFact.model_validate(json.loads(fact_json))
        result = StaleDetector().check_one(fact)
        return result.model_dump_json()

    @tool
    def filter_stale_memories(facts_json: str) -> str:
        """Filter out stale and expired memory facts from a list, returning only safe-to-use facts."""
        facts = [MemoryFact.model_validate(d) for d in json.loads(facts_json)]
        safe = StaleDetector().filter_safe(facts)
        return json.dumps([f.model_dump() for f in safe], default=str)
```

- [ ] **Step 7: Run tests**

```bash
pytest tests/test_adapters.py -v
```

Expected: PASS — 8 tests.

- [ ] **Step 8: Commit**

```bash
git add stale_detector/adapters/ tests/test_adapters.py tests/fixtures/
git commit -m "feat: add JSON, Mem0, and LangChain adapters"
```

---

### Task 7: CLI

**Files:**
- Create: `stale_detector/cli.py`

- [ ] **Step 1: Implement cli.py**

Create `stale_detector/cli.py`:
```python
import json
import sys

import click
from rich.console import Console
from rich.table import Table
from rich import box
from rich.text import Text

from stale_detector.core import StaleDetector
from stale_detector.adapters.json_adapter import load_from_json
from stale_detector.adapters.mem0_adapter import load_from_mem0
from stale_detector.models import StalenessLevel

console = Console()

_LEVEL_STYLES = {
    StalenessLevel.FRESH:   "green",
    StalenessLevel.AGING:   "yellow",
    StalenessLevel.STALE:   "red",
    StalenessLevel.EXPIRED: "bold red",
}


@click.group()
def main():
    pass


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--only-flagged", is_flag=True, help="Show only STALE and EXPIRED facts.")
@click.option("--json", "output_json", is_flag=True, help="Output raw JSON to stdout.")
@click.option("--format", "fmt", default="default",
              type=click.Choice(["default", "mem0"]), help="Input format.")
def check(file: str, only_flagged: bool, output_json: bool, fmt: str):
    """Check memory facts for staleness."""
    facts = load_from_mem0(file) if fmt == "mem0" else load_from_json(file)

    if not facts:
        click.echo("No facts found in file.")
        sys.exit(0)

    report = StaleDetector().check(facts)

    if output_json:
        click.echo(report.model_dump_json(indent=2))
        return

    rows = report.flagged if only_flagged else report.results

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("ID",           style="dim",    max_width=12)
    table.add_column("Content",                      max_width=50)
    table.add_column("Category",                     max_width=12)
    table.add_column("Age",          justify="right", max_width=6)
    table.add_column("Score",        justify="right", max_width=6)
    table.add_column("Level",                        max_width=8)
    table.add_column("Action",                       max_width=8)

    for r in rows:
        content = r.content[:47] + "..." if len(r.content) > 50 else r.content
        table.add_row(
            r.fact_id,
            content,
            r.category.value,
            str(r.age_days),
            f"{r.staleness_score:.2f}",
            Text(r.staleness_level.value.upper(), style=_LEVEL_STYLES[r.staleness_level]),
            r.recommendation,
        )

    console.print(table)
    console.print(
        f"\n[dim]Checked {report.total_facts} facts — "
        f"[green]{report.fresh_count} fresh[/], "
        f"[yellow]{report.aging_count} aging[/], "
        f"[red]{report.stale_count} stale[/], "
        f"[bold red]{report.expired_count} expired[/][/dim]"
    )
```

- [ ] **Step 2: Test the CLI with an empty file**

Create a temporary empty JSON file and run the CLI against it:
```powershell
'[]' | Out-File -FilePath "$env:TEMP\empty_test.json" -Encoding utf8
stale-detector check "$env:TEMP\empty_test.json"
```

Expected output: `No facts found in file.` — exit code 0.

- [ ] **Step 3: Commit**

```bash
git add stale_detector/cli.py
git commit -m "feat: add CLI with rich table output and empty-file guard"
```

---

### Task 8: Public API, Examples, and README

**Files:**
- Modify: `stale_detector/__init__.py`
- Create: `examples/sample_memories.json`
- Create: `examples/basic_usage.py`
- Create: `examples/langchain_integration.py`
- Create: `README.md`

- [ ] **Step 1: Populate __init__.py**

Replace `stale_detector/__init__.py` with:
```python
from stale_detector.core import StaleDetector
from stale_detector.models import (
    MemoryFact,
    StalenessResult,
    DetectionReport,
    FactCategory,
    StalenessLevel,
)

__all__ = [
    "StaleDetector",
    "MemoryFact",
    "StalenessResult",
    "DetectionReport",
    "FactCategory",
    "StalenessLevel",
]
```

- [ ] **Step 2: Create examples/sample_memories.json**

```json
[
  {
    "id": "mem_001",
    "content": "User prefers Python over JavaScript",
    "created_at": "2026-05-20T09:00:00",
    "last_confirmed_at": "2026-06-01T09:00:00",
    "confirmation_count": 3,
    "source": "user"
  },
  {
    "id": "mem_002",
    "content": "User prefers dark mode in all editors",
    "created_at": "2026-01-28T10:00:00",
    "confirmation_count": 0,
    "source": "user"
  },
  {
    "id": "mem_003",
    "content": "User is based in New Delhi for work",
    "created_at": "2025-11-19T08:00:00",
    "confirmation_count": 0,
    "source": "user"
  },
  {
    "id": "mem_004",
    "content": "User works at PwC as a senior consultant",
    "created_at": "2025-08-31T09:00:00",
    "confirmation_count": 0,
    "source": "user"
  },
  {
    "id": "mem_005",
    "content": "User is building pract-agents using LangGraph framework",
    "created_at": "2026-02-07T11:00:00",
    "confirmation_count": 0,
    "source": "user"
  },
  {
    "id": "mem_006",
    "content": "User debugged a LangGraph memory issue this morning",
    "created_at": "2026-05-08T07:00:00",
    "confirmation_count": 0,
    "source": "user"
  },
  {
    "id": "mem_007",
    "content": "User lives in Delhi",
    "created_at": "2025-05-03T10:00:00",
    "confirmation_count": 0,
    "source": "user"
  },
  {
    "id": "mem_008",
    "content": "User lives in Mumbai now",
    "created_at": "2026-04-18T10:00:00",
    "confirmation_count": 0,
    "source": "user"
  }
]
```

- [ ] **Step 3: Create examples/basic_usage.py**

```python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stale_detector import StaleDetector
from stale_detector.adapters.json_adapter import load_from_json

facts = load_from_json(os.path.join(os.path.dirname(__file__), "sample_memories.json"))
detector = StaleDetector()
report = detector.check(facts)

print(f"Total: {report.total_facts} | Flagged: {len(report.flagged)}")
for result in report.flagged:
    print(f"  [{result.staleness_level.value.upper()}] {result.content}")
    print(f"    Reason: {result.reason}")
    print(f"    Action: {result.recommendation}")
```

- [ ] **Step 4: Create examples/langchain_integration.py**

```python
"""
LangChain/LangGraph integration example.

Replace the mock invocation with a real LangGraph node in production.
Requires: pip install stale-detector[llm]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from stale_detector.adapters.langchain_tool import (
        check_memory_staleness,
        filter_stale_memories,
        LANGCHAIN_AVAILABLE,
    )
except ImportError:
    LANGCHAIN_AVAILABLE = False

if not LANGCHAIN_AVAILABLE:
    print("langchain-core not installed. Run: pip install stale-detector[llm]")
    sys.exit(0)

# --- Replace with real LangGraph node invocation in production ---
sample_fact = {
    "id": "mem_001",
    "content": "User works at PwC",
    "created_at": "2024-09-01T00:00:00",
    "confirmation_count": 0,
    "source": "user",
}

# Tool 1: check a single fact
result_json = check_memory_staleness.invoke({"fact_json": json.dumps(sample_fact)})
print("Single fact result:", result_json)

# Tool 2: filter a list — returns only FRESH and AGING facts
safe_json = filter_stale_memories.invoke({"facts_json": json.dumps([sample_fact])})
print("Safe facts:", safe_json)
```

- [ ] **Step 5: Create README.md**

Create `README.md` with exactly these 9 sections:

```markdown
# stale-detector

## What this is

`stale-detector` is a Python library and CLI that detects stale facts in an LLM agent's memory store before they are injected into the context window. It classifies each memory fact by category, computes a 0–1 staleness score based on age, confirmation history, and contradiction signals, and returns a human-readable report with recommended actions.

## The problem it solves

When an LLM agent works across sessions, it relies on stored memory facts — things like where you live, where you work, or what tech stack your project uses. These facts go stale when the real world changes but the memory doesn't. For example: a fact stored as `"User works at PwC"` remains in memory even after you change jobs. The agent retrieves it, injects it, and responds confidently with wrong information — because no tool told it the fact was outdated.

`stale-detector` proactively identifies which specific memories are at risk before they are used.

## Installation

```bash
pip install stale-detector
```

With optional LLM-assisted classification:

```bash
pip install stale-detector[llm]
```

## Quick Start

```python
from stale_detector import StaleDetector
from stale_detector.adapters.json_adapter import load_from_json

facts = load_from_json("sample_memories.json")
detector = StaleDetector()
report = detector.check(facts)

print(f"Total: {report.total_facts} | Flagged: {len(report.flagged)}")
for result in report.flagged:
    print(f"  [{result.staleness_level.value.upper()}] {result.content}")
    print(f"    Reason: {result.reason}")
    print(f"    Action: {result.recommendation}")
```

## CLI Usage

Check all facts:
```bash
stale-detector check memories.json
```

Show only stale and expired:
```bash
stale-detector check memories.json --only-flagged
```

Output raw JSON:
```bash
stale-detector check memories.json --json
```

Parse Mem0 format:
```bash
stale-detector check memories.json --format mem0
```

Sample output:
```
╭──────────┬────────────────────────────────────────┬────────────┬─────┬───────┬─────────┬─────────╮
│ ID       │ Content                                │ Category   │ Age │ Score │ Level   │ Action  │
├──────────┼────────────────────────────────────────┼────────────┼─────┼───────┼─────────┼─────────┤
│ mem_004  │ User works at PwC as a senior consu... │ employment │ 280 │  0.70 │ STALE   │ flag    │
│ mem_006  │ User debugged a LangGraph memory iss...│ episodic   │  30 │  1.00 │ EXPIRED │ discard │
╰──────────┴────────────────────────────────────────┴────────────┴─────┴───────┴─────────┴─────────╯

Checked 8 facts — 1 fresh, 2 aging, 2 stale, 3 expired
```

## Staleness Score Explained

Each fact is assigned a category with a natural lifespan:

| Category     | Examples                              | Typical Valid Window |
|--------------|---------------------------------------|----------------------|
| `location`   | "lives in Delhi", "office in Sector 5"| 6–24 months          |
| `employment` | "works at PwC", "role is consultant"  | 6–18 months          |
| `project`    | "building pract-agents", "using Pinecone" | 1–6 months       |
| `preference` | "prefers Python", "uses dark mode"    | 3–12 months          |
| `relationship`| "manager is X", "team has 5 people" | 3–12 months          |
| `identity`   | "name is X", "speaks Hindi"           | Very long/permanent  |
| `episodic`   | "debugged a LangGraph issue today"    | Days to weeks        |
| `system_fact`| "Python version is 3.10", "npm v9"   | 1–3 months           |

Score thresholds:
- `0.0 – 0.29` → **FRESH** (safe to use)
- `0.30 – 0.59` → **AGING** (use with caution)
- `0.60 – 0.79` → **STALE** (flag before injecting)
- `0.80 – 1.0` → **EXPIRED** (do not inject without reconfirmation)

## Adapters

**JSON** — default format:
```python
from stale_detector.adapters.json_adapter import load_from_json
facts = load_from_json("memories.json")
```

**Mem0** — maps `memory` → `content`, `updated_at` → `last_confirmed_at`:
```python
from stale_detector.adapters.mem0_adapter import load_from_mem0
facts = load_from_mem0("mem0_export.json")
```

**LangChain** — two tools: `check_memory_staleness` and `filter_stale_memories` (see below).

## LangChain / LangGraph Integration

```python
from stale_detector.adapters.langchain_tool import (
    check_memory_staleness,
    filter_stale_memories,
)

# In a LangGraph node — filter before injecting memories into the LLM
safe_facts_json = filter_stale_memories.invoke({"facts_json": memories_json_string})
```

Requires `pip install stale-detector[llm]`.

## Contributing

Open an issue or pull request at the project repository.
```

- [ ] **Step 6: Run the example script**

```bash
python examples/basic_usage.py
```

Expected: prints 8 facts total with flagged entries (STALE/EXPIRED) and their reasons.

- [ ] **Step 7: Run the CLI against the sample**

```bash
stale-detector check examples/sample_memories.json
```

Expected: rich table with all 8 facts, color-coded by staleness level.

- [ ] **Step 8: Run full test suite with coverage**

```bash
pytest --cov=stale_detector -v
```

Expected: All tests pass, coverage report shown.

- [ ] **Step 9: Commit**

```bash
git add stale_detector/__init__.py examples/ README.md
git commit -m "feat: public API, examples, and README"
```

---

## Self-Review

### Spec Coverage

| Spec Requirement | Covered By |
|---|---|
| `models.py` — all 5 Pydantic models with correct fields | Task 2 |
| `FactCategory` and `StalenessLevel` enums | Task 2 |
| `DetectionReport.flagged` and `.safe` properties | Task 2 |
| `model_config = ConfigDict(use_enum_values=False)` on all models | Task 2 |
| Rule-based classifier + exact `CATEGORY_KEYWORDS` table | Task 3 |
| LLM-assisted classifier + exact `CLASSIFY_PROMPT` | Task 3 |
| LLM falls back to rule-based on parse failure | Task 3 |
| Exact `DECAY_RATES` per category | Task 4 |
| Scoring formula: age × decay − confirmation_reduction | Task 4 |
| Source penalty: agent_inferred × 1.3 | Task 4 |
| Contradiction detection with anchor keyword check | Task 4 |
| +0.40 bump on contradiction | Task 4 |
| Score clamped to [0.0, 1.0] | Task 4 |
| `build_reason` — deterministic string, no LLM | Task 4 |
| `build_recommendation` — 4-level mapping | Task 4 |
| `StaleDetector.__init__` with use_llm, llm_provider, model | Task 5 |
| `StaleDetector.check()` returns `DetectionReport` | Task 5 |
| `StaleDetector.check_one()` returns `StalenessResult` | Task 5 |
| `StaleDetector.filter_safe()` returns `list[MemoryFact]` | Task 5 |
| No global mutable state | Task 5 |
| JSON adapter — required field validation + `ValueError` | Task 6 |
| Mem0 adapter — field mapping | Task 6 |
| LangChain adapter — guarded import + `LANGCHAIN_AVAILABLE` flag | Task 6 |
| Two LangChain tools with correct descriptions | Task 6 |
| CLI `check <file>` — rich table with 7 columns | Task 7 |
| CLI `--only-flagged` | Task 7 |
| CLI `--json` | Task 7 |
| CLI `--format mem0` | Task 7 |
| CLI empty file → "No facts found in file." + exit 0 | Task 7 |
| Color coding: green/yellow/red/bold-red | Task 7 |
| `sample_memories.json` — 8 facts, all levels, 1 contradiction pair | Task 8 |
| `basic_usage.py` | Task 8 |
| `langchain_integration.py` | Task 8 |
| README — all 9 sections including employer example | Task 8 |
| `.env.example` | Task 1 |
| `pyproject.toml` — exact deps, entry point | Task 1 |
| Pydantic v2 (`model_dump` / `model_validate`) used everywhere | All tasks |
| Timezone-naive datetimes, `dateutil.parser.parse().replace(tzinfo=None)` | Tasks 4, 6 |
| Forbidden deps absent (numpy, pandas, torch, etc.) | All tasks |

No gaps found. No placeholders present.
