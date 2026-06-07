# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`memlint` - Python library + CLI that detects stale facts in LLM agent memory stores before they are injected into context. Zero mandatory external services; works fully offline except for optional LLM classification.

## Commands

```bash
# Install for development
pip install -e ".[dev]"

# Install with LLM support
pip install -e ".[dev,llm]"

# Run all tests
pytest

# Run single test file
pytest tests/test_scorer.py

# Run with coverage
pytest --cov=memlint

# Run CLI
memlint check examples/sample_memories.json
memlint check examples/sample_memories.json --only-flagged
memlint check examples/sample_memories.json --json
memlint check examples/sample_memories.json --format mem0
```

## Architecture

### Module map

| File | Role |
|------|------|
| `memlint/models.py` | Pydantic v2 models: `MemoryFact`, `StalenessResult`, `DetectionReport`, enums |
| `memlint/classifier.py` | Classifies fact text → `FactCategory` via keyword matching or optional LLM call |
| `memlint/scorer.py` | Computes staleness score using decay rates, confirmation bonus, source penalty, contradiction detection |
| `memlint/core.py` | `StaleDetector` class - orchestrates classify + score for `check()`, `check_one()`, `filter_safe()` |
| `memlint/adapters/json_adapter.py` | Loads `MemoryFact` list from JSON file |
| `memlint/adapters/mem0_adapter.py` | Loads from Mem0 format (`memory` → `content`, `updated_at` → `last_confirmed_at`) |
| `memlint/adapters/langchain_tool.py` | Exposes `check_memory_staleness` and `filter_stale_memories` as LangChain tools (guarded import) |
| `memlint/cli.py` | `click` + `rich` CLI; `memlint check <file>` |

### Staleness score formula

```
score = (age_days * decay_rate) - (confirmation_count * 0.08, max 0.40)
# if source == "agent_inferred": score *= 1.3
# if contradicted by a newer fact with shared anchor keywords: score += 0.40
score = clamp(score, 0.0, 1.0)
```

Score thresholds: FRESH `[0, 0.30)` · AGING `[0.30, 0.60)` · STALE `[0.60, 0.80)` · EXPIRED `[0.80, 1.0]`

Contradiction: two facts share same category + share an anchor keyword from `CATEGORY_KEYWORDS` + were created ≥1 day apart.

### Key implementation constraints (from spec)

- **Pydantic v2 only** - use `model_dump()` / `model_validate()`, not `.dict()` / `.parse_obj()`
- **All datetimes timezone-naive** - strip tzinfo after parsing: `dt.replace(tzinfo=None)`; default `now` = `datetime.utcnow()`
- **LangChain import guard** - `langchain_tool.py` must not crash if `langchain-core` is absent; wrap in `try/except`
- **No global mutable state** - `StaleDetector` must be safely instantiatable multiple times
- **Score is deterministic** - same inputs + same `now` → same result
- **CLI handles empty file** - print `"No facts found in file."` and exit 0
- **All models** need `model_config = ConfigDict(use_enum_values=False)` so enum comparisons work
- **Forbidden deps** - do not add numpy, pandas, scikit-learn, transformers, torch, or any vector/embedding library

### LLM classifier (optional path)

Only activates when `use_llm=True` AND `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is set. Uses `langchain-openai`. If LLM response can't be parsed as `FactCategory`, falls back silently to rule-based.
