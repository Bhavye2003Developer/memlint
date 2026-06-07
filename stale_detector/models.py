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
    model_config = ConfigDict(use_enum_values=False)

    id: str
    content: str
    created_at: datetime
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
        return [r for r in self.results
                if r.staleness_level in (StalenessLevel.STALE, StalenessLevel.EXPIRED)]

    @property
    def safe(self) -> list[StalenessResult]:
        return [r for r in self.results
                if r.staleness_level in (StalenessLevel.FRESH, StalenessLevel.AGING)]
