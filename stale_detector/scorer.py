from datetime import datetime
from stale_detector.models import FactCategory, MemoryFact, StalenessLevel
from stale_detector.classifier import CATEGORY_KEYWORDS

DECAY_RATES: dict[FactCategory, float] = {
    FactCategory.LOCATION:      0.0020,
    FactCategory.EMPLOYMENT:    0.0025,
    FactCategory.PROJECT:       0.0060,
    FactCategory.PREFERENCE:    0.0030,
    FactCategory.RELATIONSHIP:  0.0025,
    FactCategory.IDENTITY:      0.0005,
    FactCategory.EPISODIC:      0.0500,
    FactCategory.SYSTEM_FACT:   0.0100,
    FactCategory.UNKNOWN:       0.0030,
}


def determine_level(score: float) -> StalenessLevel:
    if score < 0.30:
        return StalenessLevel.FRESH
    if score < 0.60:
        return StalenessLevel.AGING
    if score < 0.80:
        return StalenessLevel.STALE
    return StalenessLevel.EXPIRED


def build_reason(
    age_days: int,
    category: FactCategory,
    confirmation_count: int,
    has_contradiction: bool,
    score: float,
) -> str:
    decay_days = int(1 / DECAY_RATES[category])
    parts = [f"{age_days} days old ({category.value} facts decay in ~{decay_days} days)"]
    if confirmation_count > 0:
        parts.append(f"confirmed {confirmation_count} time(s)")
    if has_contradiction:
        parts.append("a newer conflicting fact exists")
    return "; ".join(parts)


def build_recommendation(level: StalenessLevel) -> str:
    return {
        StalenessLevel.FRESH:   "use",
        StalenessLevel.AGING:   "verify",
        StalenessLevel.STALE:   "flag",
        StalenessLevel.EXPIRED: "discard",
    }[level]


def _are_contradictory(
    fact_a: MemoryFact,
    fact_b: MemoryFact,
    category: FactCategory,
) -> bool:
    if fact_a.category != fact_b.category and not (
        fact_a.category == category or fact_b.category == category
    ):
        return False

    time_diff = abs((fact_a.created_at - fact_b.created_at).days)
    if time_diff < 1:
        return False

    keywords = CATEGORY_KEYWORDS.get(category, [])
    a_lower = fact_a.content.lower()
    b_lower = fact_b.content.lower()
    return any(kw in a_lower and kw in b_lower for kw in keywords)


def compute_staleness_score(
    fact: MemoryFact,
    category: FactCategory,
    all_facts: list[MemoryFact],
    now: datetime,
) -> tuple[float, bool, str | None]:
    """Returns (score, has_contradiction, contradicted_by_id)."""
    reference_time = fact.last_confirmed_at or fact.created_at
    age_days = max((now - reference_time).days, 0)

    decay_rate = DECAY_RATES[category]
    score = age_days * decay_rate

    confirmation_reduction = min(fact.confirmation_count * 0.08, 0.40)
    score -= confirmation_reduction

    if fact.source == "agent_inferred":
        score *= 1.3

    has_contradiction = False
    contradicted_by: str | None = None
    for other in all_facts:
        if other.id == fact.id:
            continue
        if _are_contradictory(fact, other, category):
            has_contradiction = True
            contradicted_by = other.id
            score += 0.40
            break

    return min(max(score, 0.0), 1.0), has_contradiction, contradicted_by
