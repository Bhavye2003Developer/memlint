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
