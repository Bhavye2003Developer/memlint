import json
from memlint.models import MemoryFact

try:
    from langchain_core.tools import tool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

if LANGCHAIN_AVAILABLE:
    from memlint.core import StaleDetector

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
