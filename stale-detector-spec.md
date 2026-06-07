# Stale Detector — Knowledge Primer + Build Spec

---

## PART 1: KNOWLEDGE PRIMER (Read This First)

### What is LLM Memory?

When an LLM agent works across sessions, it needs to "remember" things about the user or world. This memory is stored externally — usually as a list of plain text facts in a file, database, or vector store. Examples:

```
"User lives in Delhi"
"User works at PwC as a risk consultant"
"User prefers Python over JavaScript"
"User's project is called pract-agents"
```

These facts get retrieved and injected into the LLM's context window before each response, so the agent can personalize and reason correctly.

### What is Memory Staleness?

A memory becomes **stale** when the real-world fact it describes has changed, but the stored memory has not been updated. The agent then acts on wrong information — confidently.

**Real examples:**
- Stored: `"User lives in Delhi"` → User moved to Mumbai 3 months ago
- Stored: `"User works at PwC"` → User changed jobs
- Stored: `"User's project uses Python 3.10"` → Project now uses 3.12
- Stored: `"User prefers dark mode"` → Changed preference

The dangerous part: the agent doesn't know the memory is wrong. It retrieves it, injects it, and responds as if it's truth.

### Why Current Tools Don't Solve This

Current memory systems (Mem0, LangGraph checkpointing, file-based stores) use **recency-based decay** — they softly downrank older memories at retrieval time. This handles low-stakes staleness (old breakfast preference). It does NOT handle high-stakes staleness (wrong employer, wrong location, outdated project state) because those memories are still highly semantically relevant and get retrieved at full strength.

No tool today proactively tells you *which specific memories are at risk of being wrong* before the agent uses them.

### How Staleness Detection Works (The Core Idea)

Each memory fact has properties that determine how fast it goes stale:

**1. Fact Category** — Different facts have different natural lifespans:

| Category | Examples | Typical Valid Window |
|---|---|---|
| `location` | "lives in Delhi", "office is in Sector 5" | 6–24 months |
| `employment` | "works at PwC", "role is consultant" | 6–18 months |
| `project` | "building pract-agents", "using Pinecone" | 1–6 months |
| `preference` | "prefers Python", "uses dark mode" | 3–12 months |
| `relationship` | "manager is X", "team has 5 people" | 3–12 months |
| `identity` | "name is X", "speaks Hindi" | Very long / permanent |
| `episodic` | "debugged a LangGraph issue today" | Days to weeks |
| `system_fact` | "Python version is 3.10", "using npm v9" | 1–3 months |

**2. Age** — How many days since this fact was written or last confirmed.

**3. Confirmation Count** — Has the user mentioned this fact again recently? Each reconfirmation resets the staleness clock.

**4. Contradiction Signal** — Has a newer memory been stored that contradicts this one? (e.g., new location stored while old one still exists)

**5. Staleness Score Formula (simplified):**

```
base_decay_rate = decay_rates[category]         # e.g., 0.003 per day for employment
age_days = (now - created_at).days
base_score = age_days * base_decay_rate         # 0.0 to 1.0+

# Reduce if recently confirmed
recency_bonus = confirmation_count * 0.1
score = base_score - recency_bonus

# Boost if contradicted
if has_contradiction:
    score += 0.4

# Clamp to [0.0, 1.0]
staleness_score = min(max(score, 0.0), 1.0)
```

**Score interpretation:**
- `0.0 – 0.3` → FRESH (safe to use)
- `0.3 – 0.6` → AGING (use with caution)  
- `0.6 – 0.8` → STALE (flag before injecting)
- `0.8 – 1.0` → EXPIRED (do not inject without reconfirmation)

### What the Tool Does

`stale-detector` is a **Python library** that:

1. Accepts a list of memory facts (plain text strings + metadata)
2. Classifies each fact into a category using a lightweight LLM call
3. Computes a staleness score for each fact
4. Returns scores + human-readable reasons + recommended action
5. Can be used as a LangChain tool, called from a LangGraph node, or used standalone via CLI

---

## PART 2: BUILD SPEC

> This spec is written for Claude Code. Follow it exactly. Do not deviate from the architecture described. Do not add features not listed here. Do not use libraries not listed here.

---

### Project Overview

**Name:** `stale-detector`  
**Type:** Python library + CLI  
**Language:** Python 3.11+  
**Purpose:** Detect stale facts in an LLM agent's memory store before they are injected into context  
**Key constraint:** Zero mandatory external services. Works fully offline except for the optional LLM classification call.

---

### Directory Structure

```
stale-detector/
├── stale_detector/
│   ├── __init__.py
│   ├── core.py            # Main StaleDetector class
│   ├── classifier.py      # Fact category classification (rule-based + optional LLM)
│   ├── scorer.py          # Staleness score computation
│   ├── models.py          # Pydantic data models
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── json_adapter.py     # Load memories from JSON file
│   │   ├── mem0_adapter.py     # Load memories from Mem0 format
│   │   └── langchain_tool.py   # Wrap as LangChain tool
│   └── cli.py             # CLI entry point
├── tests/
│   ├── test_classifier.py
│   ├── test_scorer.py
│   └── test_core.py
├── examples/
│   ├── basic_usage.py
│   ├── langchain_integration.py
│   └── sample_memories.json
├── pyproject.toml
├── README.md
└── .env.example
```

---

### Dependencies

**`pyproject.toml`** — use these exact versions:

```toml
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

**Do NOT add:** numpy, pandas, scikit-learn, transformers, torch, or any vector/embedding library. This tool must be lightweight.

---

### Data Models (`models.py`)

```python
from pydantic import BaseModel, Field
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
    FRESH = "fresh"        # score 0.0 – 0.29
    AGING = "aging"        # score 0.30 – 0.59
    STALE = "stale"        # score 0.60 – 0.79
    EXPIRED = "expired"    # score 0.80 – 1.0


class MemoryFact(BaseModel):
    id: str                                         # unique identifier
    content: str                                    # the raw text fact
    created_at: datetime                            # when it was first stored
    last_confirmed_at: Optional[datetime] = None    # last time user re-stated this
    confirmation_count: int = 0                     # how many times re-confirmed
    category: Optional[FactCategory] = None         # auto-classified if None
    source: str = "user"                            # "user" | "agent_inferred"
    metadata: dict = Field(default_factory=dict)    # arbitrary extra fields


class StalenessResult(BaseModel):
    fact_id: str
    content: str
    category: FactCategory
    staleness_score: float          # 0.0 to 1.0
    staleness_level: StalenessLevel
    age_days: int
    reason: str                     # human-readable explanation
    recommendation: str             # "use" | "verify" | "flag" | "discard"
    has_contradiction: bool = False
    contradicted_by: Optional[str] = None  # id of the contradicting fact


class DetectionReport(BaseModel):
    checked_at: datetime
    total_facts: int
    fresh_count: int
    aging_count: int
    stale_count: int
    expired_count: int
    results: list[StalenessResult]

    @property
    def flagged(self) -> list[StalenessResult]:
        """Facts that are stale or expired."""
        return [r for r in self.results if r.staleness_level in (StalenessLevel.STALE, StalenessLevel.EXPIRED)]

    @property
    def safe(self) -> list[StalenessResult]:
        """Facts that are fresh or aging."""
        return [r for r in self.results if r.staleness_level in (StalenessLevel.FRESH, StalenessLevel.AGING)]
```

---

### Classifier (`classifier.py`)

Two modes: **rule-based** (default, no LLM needed) and **LLM-assisted** (optional, better accuracy).

#### Rule-Based Classification

Use keyword matching. Map fact text to category using this exact keyword table:

```python
CATEGORY_KEYWORDS: dict[FactCategory, list[str]] = {
    FactCategory.LOCATION: [
        "lives", "located", "based in", "address", "city", "country",
        "office", "moved to", "residing", "hometown", "location"
    ],
    FactCategory.EMPLOYMENT: [
        "works at", "employed", "job", "role", "position", "company",
        "organization", "joined", "hired", "manager", "team", "department",
        "title", "consultant", "engineer", "analyst", "intern"
    ],
    FactCategory.PROJECT: [
        "project", "building", "repo", "codebase", "app", "tool",
        "working on", "developing", "implementing", "stack", "framework",
        "library", "version", "api", "endpoint", "deployed", "launched"
    ],
    FactCategory.PREFERENCE: [
        "prefers", "likes", "favorite", "enjoys", "uses", "dislikes",
        "wants", "chooses", "opts for", "theme", "mode", "setting",
        "style", "approach"
    ],
    FactCategory.RELATIONSHIP: [
        "friend", "colleague", "manager", "reports to", "partner",
        "teammate", "mentor", "client", "collaborator", "family"
    ],
    FactCategory.IDENTITY: [
        "name is", "called", "age", "born", "nationality", "speaks",
        "gender", "education", "degree", "graduated", "alumni"
    ],
    FactCategory.EPISODIC: [
        "today", "yesterday", "last week", "this morning", "just",
        "recently", "earlier", "said that", "mentioned", "asked about",
        "discussed", "fixed", "resolved", "debugging"
    ],
    FactCategory.SYSTEM_FACT: [
        "python version", "node version", "npm", "pip", "docker",
        "os", "operating system", "machine", "cpu", "ram", "disk",
        "installed", "configured", "environment", "env", ".env"
    ],
}
```

Logic: lowercase the fact content, check which category has the most keyword hits, assign that category. If no keywords match, return `UNKNOWN`.

#### LLM-Assisted Classification (optional)

Only used when `use_llm=True` is passed AND `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`) is set. Use `langchain-openai` with this exact prompt:

```python
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
```

Parse the response as a `FactCategory` enum value. If parsing fails, fall back to rule-based.

---

### Decay Rates (`scorer.py`)

These are the exact decay rates per category (how much staleness score increases per day):

```python
DECAY_RATES: dict[FactCategory, float] = {
    FactCategory.LOCATION:      0.0020,   # fully stale after ~500 days
    FactCategory.EMPLOYMENT:    0.0025,   # fully stale after ~400 days
    FactCategory.PROJECT:       0.0060,   # fully stale after ~167 days
    FactCategory.PREFERENCE:    0.0030,   # fully stale after ~333 days
    FactCategory.RELATIONSHIP:  0.0025,   # fully stale after ~400 days
    FactCategory.IDENTITY:      0.0005,   # fully stale after ~2000 days (very stable)
    FactCategory.EPISODIC:      0.0500,   # fully stale after ~20 days (fast decay)
    FactCategory.SYSTEM_FACT:   0.0100,   # fully stale after ~100 days
    FactCategory.UNKNOWN:       0.0030,   # default
}

STALENESS_THRESHOLDS = {
    "fresh":   (0.00, 0.30),
    "aging":   (0.30, 0.60),
    "stale":   (0.60, 0.80),
    "expired": (0.80, 1.00),
}
```

#### Scoring Logic (implement exactly as described):

```python
def compute_staleness_score(
    fact: MemoryFact,
    category: FactCategory,
    all_facts: list[MemoryFact],
    now: datetime,
) -> tuple[float, bool, str | None]:
    """
    Returns: (score, has_contradiction, contradicted_by_id)
    """
    # 1. Age in days
    reference_time = fact.last_confirmed_at or fact.created_at
    age_days = max((now - reference_time).days, 0)

    # 2. Base score from decay
    decay_rate = DECAY_RATES[category]
    base_score = age_days * decay_rate

    # 3. Confirmation bonus (each confirmation reduces score by 0.08, max 0.4 reduction)
    confirmation_reduction = min(fact.confirmation_count * 0.08, 0.40)
    score = base_score - confirmation_reduction

    # 4. Source penalty: agent-inferred facts decay faster (multiply base by 1.3)
    if fact.source == "agent_inferred":
        score *= 1.3

    # 5. Contradiction detection
    has_contradiction = False
    contradicted_by = None
    for other in all_facts:
        if other.id == fact.id:
            continue
        if _are_contradictory(fact, other, category):
            has_contradiction = True
            contradicted_by = other.id
            score += 0.40  # significant bump when contradicted
            break

    # 6. Clamp to [0.0, 1.0]
    score = min(max(score, 0.0), 1.0)

    return score, has_contradiction, contradicted_by
```

#### Contradiction Detection

Two facts are contradictory if:
1. They share the same category AND
2. One was created strictly AFTER the other AND
3. They share at least one "anchor keyword" from the category's keyword list (meaning they're about the same topic, not just the same type)

```python
def _are_contradictory(fact_a: MemoryFact, fact_b: MemoryFact, category: FactCategory) -> bool:
    if fact_a.category != fact_b.category and not (
        fact_a.category == category or fact_b.category == category
    ):
        return False

    # One must be newer than the other (at least 1 day apart)
    time_diff = abs((fact_a.created_at - fact_b.created_at).days)
    if time_diff < 1:
        return False

    # They must share anchor keywords (same topic)
    keywords = CATEGORY_KEYWORDS.get(category, [])
    a_lower = fact_a.content.lower()
    b_lower = fact_b.content.lower()
    shared = any(kw in a_lower and kw in b_lower for kw in keywords)
    return shared
```

#### Reason and Recommendation Generation

Generate human-readable reason strings based on the score components. Do NOT call an LLM for this — use deterministic string formatting:

```python
def build_reason(age_days, category, confirmation_count, has_contradiction, score) -> str:
    parts = []
    parts.append(f"{age_days} days old ({category.value} facts decay in ~{int(1/DECAY_RATES[category])} days)")
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
```

---

### Core Class (`core.py`)

```python
class StaleDetector:
    def __init__(
        self,
        use_llm: bool = False,
        llm_provider: str = "openai",   # "openai" | "anthropic"
        model: str = "gpt-4o-mini",
    ):
        ...

    def check(
        self,
        facts: list[MemoryFact],
        now: datetime | None = None,
    ) -> DetectionReport:
        """
        Main entry point. Check all facts and return a full report.
        - Classifies each fact (rule-based or LLM)
        - Computes staleness score for each
        - Detects contradictions across all facts
        - Returns DetectionReport
        """
        ...

    def check_one(
        self,
        fact: MemoryFact,
        context_facts: list[MemoryFact] | None = None,
        now: datetime | None = None,
    ) -> StalenessResult:
        """
        Check a single fact. context_facts used for contradiction detection.
        """
        ...

    def filter_safe(
        self,
        facts: list[MemoryFact],
        now: datetime | None = None,
    ) -> list[MemoryFact]:
        """
        Returns only FRESH and AGING facts. Convenience method for
        pre-filtering before injecting into LLM context.
        """
        report = self.check(facts, now)
        safe_ids = {r.fact_id for r in report.safe}
        return [f for f in facts if f.id in safe_ids]
```

---

### JSON Adapter (`adapters/json_adapter.py`)

Must be able to load memories from a JSON file in this format:

```json
[
  {
    "id": "mem_001",
    "content": "User lives in Delhi",
    "created_at": "2024-08-15T10:00:00",
    "last_confirmed_at": null,
    "confirmation_count": 0,
    "source": "user"
  }
]
```

The adapter must:
- Load the JSON file
- Parse each entry into a `MemoryFact` object
- Handle missing optional fields gracefully with defaults
- Raise a clear `ValueError` with message if required fields (`id`, `content`, `created_at`) are missing

---

### Mem0 Adapter (`adapters/mem0_adapter.py`)

Mem0 stores memories in this format:
```json
{
  "id": "abc123",
  "memory": "User works at PwC",
  "created_at": "2024-09-01T00:00:00",
  "updated_at": "2024-09-01T00:00:00",
  "metadata": {}
}
```

Map fields:
- `memory` → `content`
- `updated_at` → `last_confirmed_at`
- All other fields map directly
- `confirmation_count` defaults to 0
- `source` defaults to `"user"`

---

### LangChain Tool (`adapters/langchain_tool.py`)

Only created if `langchain-core` is installed. Guard the import:

```python
try:
    from langchain_core.tools import tool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
```

Expose two tools:

**Tool 1: `check_memory_staleness`**
- Input: JSON string of a single memory fact
- Output: JSON string of StalenessResult
- Description: "Check if a single memory fact is stale before injecting it into context."

**Tool 2: `filter_stale_memories`**
- Input: JSON string array of memory facts
- Output: JSON string array of only FRESH and AGING facts
- Description: "Filter out stale and expired memory facts from a list, returning only safe-to-use facts."

---

### CLI (`cli.py`)

Use `click` and `rich` for output. Commands:

#### `stale-detector check <file>`

```bash
stale-detector check memories.json
```

Output a rich table with columns:
`ID | Content (truncated to 50 chars) | Category | Age | Score | Level | Recommendation`

Color coding:
- FRESH → green
- AGING → yellow
- STALE → red
- EXPIRED → bold red

#### `stale-detector check <file> --only-flagged`

Only show STALE and EXPIRED rows.

#### `stale-detector check <file> --json`

Output raw JSON of the DetectionReport to stdout. No rich formatting.

#### `stale-detector check <file> --format mem0`

Parse input as Mem0 format instead of default JSON format.

---

### Example Files (`examples/`)

#### `examples/sample_memories.json`

Include exactly these 8 facts with realistic `created_at` dates spread across the past 2 years. Include at least:
- 1 FRESH fact (created recently, confirmed multiple times)
- 2 AGING facts
- 2 STALE facts
- 1 EXPIRED fact (old episodic)
- 1 pair of contradictory facts (same category, different values, older and newer)

#### `examples/basic_usage.py`

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

#### `examples/langchain_integration.py`

Show how to use the LangChain tools inside a LangGraph node. Use a placeholder LLM (no real API calls in the example). Show the import, tool initialization, and a mock invocation with a comment explaining what to replace.

---

### Tests (`tests/`)

#### `test_classifier.py`
- Test that `"User lives in Delhi"` classifies as `LOCATION`
- Test that `"User works at PwC as a consultant"` classifies as `EMPLOYMENT`
- Test that `"Building pract-agents using LangGraph"` classifies as `PROJECT`
- Test that `"User prefers Python over JavaScript"` classifies as `PREFERENCE`
- Test UNKNOWN fallback for `"The sky is blue"`

#### `test_scorer.py`
- Test FRESH: fact created 10 days ago, category IDENTITY → score < 0.30
- Test EXPIRED: fact created 30 days ago, category EPISODIC → score > 0.80
- Test confirmation reduces score: same fact with `confirmation_count=5` has lower score than `confirmation_count=0`
- Test contradiction: two LOCATION facts where one is newer → older one has `has_contradiction=True`
- Test agent_inferred source: same fact ages faster than user-sourced

#### `test_core.py`
- Test `check()` returns a `DetectionReport` with correct counts
- Test `filter_safe()` excludes STALE and EXPIRED facts
- Test `check_one()` on a single fact returns a `StalenessResult`

---

### README.md

Must contain exactly these sections:

1. **What this is** — one paragraph, no fluff
2. **The problem it solves** — explain staleness in plain English, with the employer example
3. **Installation** — `pip install stale-detector` and `pip install stale-detector[llm]`
4. **Quick Start** — the basic_usage.py example verbatim
5. **CLI Usage** — show the three CLI commands with sample output
6. **Staleness Score Explained** — the category table and score thresholds
7. **Adapters** — JSON, Mem0, LangChain
8. **LangChain / LangGraph Integration** — code snippet
9. **Contributing** — one sentence pointing to issues

---

### `.env.example`

```
# Optional: only needed if use_llm=True
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

---

## PART 3: IMPLEMENTATION RULES FOR CLAUDE CODE

1. **No hallucinated libraries.** Only use what is listed in `pyproject.toml`. Do not import anything else.

2. **No placeholder functions.** Every function listed in this spec must be fully implemented, not stubbed with `pass` or `raise NotImplementedError`.

3. **Pydantic v2 only.** Use `model_dump()`, not `.dict()`. Use `model_validate()`, not `.parse_obj()`.

4. **datetime handling.** All datetimes must be timezone-naive (no `tzinfo`). Use `datetime.utcnow()` as the default `now`. When parsing from strings, use `dateutil.parser.parse()` and strip tzinfo: `dt.replace(tzinfo=None)`.

5. **LLM import guard.** The `langchain_tool.py` adapter must not crash when `langchain-core` is not installed. Wrap all langchain imports in try/except as shown.

6. **No global state.** `StaleDetector` must be instantiatable multiple times without side effects. No module-level mutable state.

7. **Score must be deterministic.** Given the same inputs and the same `now`, `compute_staleness_score` must always return the same result.

8. **CLI must not crash on empty file.** If the input JSON file is empty or contains 0 facts, print `"No facts found in file."` and exit with code 0.

9. **All Pydantic models must have `model_config = ConfigDict(use_enum_values=False)`** so enum comparisons work correctly.

10. **Tests must be runnable with `pytest` from the project root with no configuration.** No fixtures requiring external services.
