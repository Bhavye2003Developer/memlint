# memlint

**Lint your LLM agent's memory before it lies to you.**

`memlint` detects stale facts in an LLM agent's memory store before they are injected into the context window. It scores each fact by age, confirmation history, and contradiction signals, then tells you which ones to flag, refresh, or discard.

## The problem

LLM agents that work across sessions store facts about the user and world - where they live, where they work, what they're building. These facts go stale when the real world changes but the memory doesn't. A fact like `"User works at xyz"` stays in memory after a job change. The agent retrieves it, injects it, and answers confidently with wrong information.

`memlint` catches this before it happens.

## Installation

```bash
pip install memlint
```

With optional LLM-assisted classification:

```bash
pip install memlint[llm]
```

## Quick Start

```python
from memlint import StaleDetector
from memlint.adapters.json_adapter import load_from_json

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
memlint check memories.json
```

Show only stale and expired:
```bash
memlint check memories.json --only-flagged
```

Output raw JSON:
```bash
memlint check memories.json --json
```

Parse Mem0 format:
```bash
memlint check memories.json --format mem0
```

Sample output:
```
╭──────────┬────────────────────────────────────────┬────────────┬─────┬───────┬─────────┬─────────╮
│ ID       │ Content                                │ Category   │ Age │ Score │ Level   │ Action  │
├──────────┼────────────────────────────────────────┼────────────┼─────┼───────┼─────────┼─────────┤
│ mem_004  │ User works at XYZ as a senior cons...  │ employment │ 279 │  0.70 │ STALE   │ flag    │
│ mem_006  │ User debugged a LangGraph memory is... │ episodic   │  29 │  1.00 │ EXPIRED │ discard │
╰──────────┴────────────────────────────────────────┴────────────┴─────┴───────┴─────────┴─────────╯

Checked 8 facts: 1 fresh, 2 aging, 3 stale, 2 expired
```

## Staleness Score Explained

Each fact is assigned a category with a natural lifespan:

| Category     | Examples                              | Typical Valid Window |
|--------------|---------------------------------------|----------------------|
| `location`   | "lives in Delhi", "office in Sector 5"| 6–24 months          |
| `employment` | "works at xyz", "role is consultant"  | 6–18 months          |
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

**JSON**: default format:
```python
from memlint.adapters.json_adapter import load_from_json
facts = load_from_json("memories.json")
```

**Mem0**: maps `memory` to `content`, `updated_at` to `last_confirmed_at`:
```python
from memlint.adapters.mem0_adapter import load_from_mem0
facts = load_from_mem0("mem0_export.json")
```

**LangChain**: two tools: `check_memory_staleness` and `filter_stale_memories` (see below).

## LangChain / LangGraph Integration

```python
from memlint.adapters.langchain_tool import (
    check_memory_staleness,
    filter_stale_memories,
)

# In a LangGraph node: filter before injecting memories into the LLM
safe_facts_json = filter_stale_memories.invoke({"facts_json": memories_json_string})
```

Requires `pip install memlint[llm]`.

## Contributing

Open an issue or pull request at the project repository.
