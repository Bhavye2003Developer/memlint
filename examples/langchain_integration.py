"""
LangChain/LangGraph integration example.

Replace the mock invocation with a real LangGraph node in production.
Requires: pip install memlint[llm]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from memlint.adapters.langchain_tool import (
        check_memory_staleness,
        filter_stale_memories,
        LANGCHAIN_AVAILABLE,
    )
except ImportError:
    LANGCHAIN_AVAILABLE = False

if not LANGCHAIN_AVAILABLE:
    print("langchain-core not installed. Run: pip install memlint[llm]")
    sys.exit(0)

# --- Replace with real LangGraph node invocation in production ---
sample_fact = {
    "id": "mem_001",
    "content": "User works at Acme Corp",
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
