import warnings
from datetime import datetime


def create_memory_metadata(
    created_at: datetime | None = None,
    source: str = "user",
    confirmation_count: int = 0,
) -> dict:
    """Generate a metadata dict to store alongside vectors or embeddings in your DB.

    The returned dict can be passed directly as ``**metadata`` to ``MemoryFact``
    when loading memories for staleness checking.

    Args:
        created_at: When this memory was created. If omitted, falls back to the
            current UTC time and emits a warning. Pass the real timestamp for
            accurate staleness scoring.
        source: ``"user"`` or ``"agent_inferred"``. Agent-inferred facts receive
            a 1.3x staleness penalty.
        confirmation_count: Number of times this fact has been reconfirmed.

    Returns:
        Dict with keys ``created_at`` (ISO string), ``source``, ``confirmation_count``.

    Example::

        # At embedding time — store this alongside your vector
        metadata = create_memory_metadata(created_at=datetime(2024, 6, 1))
        collection.upsert(id="mem_001", vector=embedding, metadata=metadata)

        # At retrieval time — load directly into MemoryFact
        fact = MemoryFact(id=doc["id"], content=doc["text"], **doc["metadata"])
    """
    if created_at is None:
        warnings.warn(
            "created_at not provided — using current UTC time as fallback. "
            "For accurate staleness scoring, pass the original creation timestamp.",
            UserWarning,
            stacklevel=2,
        )
        created_at = datetime.utcnow()

    return {
        "created_at": created_at.isoformat(),
        "source": source,
        "confirmation_count": confirmation_count,
    }
