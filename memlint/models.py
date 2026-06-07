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
    """A single memory fact stored in an LLM agent's memory store.

    Attributes:
        id: Unique identifier for this fact.
        content: The plain-text content of the memory.
        created_at: When this fact was first stored (timezone-naive UTC).
            Defaults to current UTC time if not provided.
        last_confirmed_at: Last time the fact was reconfirmed. Resets decay clock.
        confirmation_count: How many times the fact has been reconfirmed.
        category: Optional pre-assigned category. Skips classification if set.
        source: Origin of the fact, either ``"user"`` or ``"agent_inferred"``.
            Agent-inferred facts receive a 1.3x staleness penalty.
        metadata: Arbitrary extra data passed through unchanged.
    """
    model_config = ConfigDict(use_enum_values=False)

    id: str
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
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
    """Full staleness report for a batch of memory facts.

    Use ``flagged`` to get facts that should not be injected into context.
    Use ``safe`` to get facts that are fresh enough to use.
    """
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
        """STALE and EXPIRED facts. Do not inject these into LLM context."""
        return [r for r in self.results
                if r.staleness_level in (StalenessLevel.STALE, StalenessLevel.EXPIRED)]

    @property
    def safe(self) -> list[StalenessResult]:
        """FRESH and AGING facts, safe to inject into LLM context."""
        return [r for r in self.results
                if r.staleness_level in (StalenessLevel.FRESH, StalenessLevel.AGING)]
