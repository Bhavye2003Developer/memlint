from __future__ import annotations

import warnings
from datetime import datetime

from memlint.models import MemoryFact


def create_memory_metadata(
    created_at: datetime | None = None,
    source: str = "user",
    confirmation_count: int = 0,
    current_default: bool = False,
) -> dict:
    """Generate a metadata dict to store alongside vectors or embeddings in your DB.

    The returned dict can be passed directly as ``**metadata`` to ``MemoryFact``
    when loading memories for staleness checking.

    Args:
        created_at: When this memory was created. If omitted and
            ``current_default`` is False, falls back to the current UTC time
            and emits a warning. Pass the real timestamp for accurate staleness
            scoring.
        source: ``"user"`` or ``"agent_inferred"``. Agent-inferred facts receive
            a 1.3x staleness penalty.
        confirmation_count: Number of times this fact has been reconfirmed.
        current_default: If True, use the current UTC time without emitting a
            warning. Useful when you genuinely want now as the creation time and
            do not want to import datetime yourself.

    Returns:
        Dict with keys ``created_at`` (ISO string), ``source``, ``confirmation_count``.

    Example::

        # No datetime import needed
        metadata = create_memory_metadata(current_default=True)
        collection.upsert(id="mem_001", vector=embedding, metadata=metadata)

        # At retrieval time, load directly into MemoryFact
        fact = MemoryFact(id=doc["id"], content=doc["text"], **doc["metadata"])
    """
    if current_default:
        created_at = datetime.utcnow()
    elif created_at is None:
        warnings.warn(
            "created_at not provided, using current UTC time as fallback. "
            "For accurate staleness scoring, pass the original creation timestamp "
            "or set current_default=True to suppress this warning.",
            UserWarning,
            stacklevel=2,
        )
        created_at = datetime.utcnow()

    return {
        "created_at": created_at.isoformat(),
        "source": source,
        "confirmation_count": confirmation_count,
    }


def confirm_fact(fact: MemoryFact, now: datetime | None = None) -> MemoryFact:
    """Return a copy of the fact with confirmation count incremented and last_confirmed_at updated.

    Does not mutate the original fact.

    Args:
        fact: The fact to confirm.
        now: Timestamp to use as ``last_confirmed_at``. Defaults to ``datetime.utcnow()``.

    Example::

        updated = confirm_fact(fact)
        # store updated fact back to your DB
    """
    if now is None:
        now = datetime.utcnow()
    return fact.model_copy(update={
        "confirmation_count": fact.confirmation_count + 1,
        "last_confirmed_at": now,
    })


def confirm_facts(facts: list[MemoryFact], now: datetime | None = None) -> list[MemoryFact]:
    """Return a list of confirmed copies of the given facts.

    Equivalent to calling ``confirm_fact`` on each fact. Does not mutate originals.

    Args:
        facts: Facts to confirm.
        now: Shared timestamp for all confirmations. Defaults to ``datetime.utcnow()``.
    """
    if now is None:
        now = datetime.utcnow()
    return [confirm_fact(f, now) for f in facts]
