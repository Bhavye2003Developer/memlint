import json
from memlint.adapters._utils import parse_dt
from memlint.models import MemoryFact


def load_from_mem0(filepath: str) -> list[MemoryFact]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array at root, got {type(data).__name__}")

    facts = []
    for i, entry in enumerate(data):
        for required in ("id", "memory", "created_at"):
            if required not in entry:
                raise ValueError(f"Entry {i} missing required field '{required}'")

        facts.append(MemoryFact(
            id=entry["id"],
            content=entry["memory"],
            created_at=parse_dt(entry["created_at"]),
            last_confirmed_at=parse_dt(entry.get("updated_at")),
            confirmation_count=entry.get("confirmation_count", 0),
            source=entry.get("source", "user"),
            metadata=entry.get("metadata", {}),
        ))
    return facts
