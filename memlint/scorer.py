from datetime import datetime, timedelta
from memlint.models import FactCategory, MemoryFact, StalenessLevel
from memlint.classifier import CATEGORY_KEYWORDS

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

NEGATION_SIGNALS: list[str] = [
    "no longer", "left ", "quit", "resigned", "moved out", "moved from",
    "former ", "used to", "not anymore", "switched to", "stopped working",
    "retired", "fired", "laid off", "ended", "closed down", "shut down",
    "relocated from", "departed", "dropped out",
]

TRANSITION_KEYWORDS: dict[FactCategory, list[str]] = {
    FactCategory.EMPLOYMENT: [
        "left", "quit", "resigned", "fired", "laid off", "former", "moved on", "switched jobs",
    ],
    FactCategory.LOCATION: [
        "moved", "relocated", "moved out", "moved from", "left",
    ],
    FactCategory.PROJECT: [
        "dropped", "cancelled", "abandoned", "migrated from", "replaced", "deprecated", "switched from",
    ],
    FactCategory.PREFERENCE: [
        "switched from", "no longer uses", "stopped using", "replaced",
    ],
    FactCategory.SYSTEM_FACT: [
        "upgraded from", "migrated from", "replaced", "uninstalled", "switched from",
    ],
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


_COMMON_WORDS = {"user", "the", "this", "that", "they", "their", "has", "have", "had"}


def _extract_proper_nouns(text: str) -> set[str]:
    return {
        w.strip('.,!?;"\'').lower()
        for w in text.split()
        if w and w[0].isupper() and len(w.strip('.,!?;"\'')) > 2
        and w.strip('.,!?;"\'').lower() not in _COMMON_WORDS
    }


def _has_negation_signal(text: str) -> bool:
    lower = text.lower()
    return any(signal in lower for signal in NEGATION_SIGNALS)


def _has_transition_keyword(text: str, category: FactCategory) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in TRANSITION_KEYWORDS.get(category, []))


def _are_contradictory(
    fact_a: MemoryFact,
    fact_b: MemoryFact,
    category: FactCategory,
) -> bool:
    if fact_a.category != fact_b.category and not (
        fact_a.category == category or fact_b.category == category
    ):
        return False

    time_diff = abs(fact_a.created_at - fact_b.created_at)
    if time_diff < timedelta(days=1):
        return False

    a_lower = fact_a.content.lower()
    b_lower = fact_b.content.lower()
    keywords = CATEGORY_KEYWORDS.get(category, [])

    # existing: shared anchor keyword
    if any(kw in a_lower and kw in b_lower for kw in keywords):
        return True

    # semantic: shared proper noun + negation signal in either fact
    a_entities = _extract_proper_nouns(fact_a.content)
    b_entities = _extract_proper_nouns(fact_b.content)
    shared_entities = a_entities & b_entities

    if shared_entities:
        if _has_negation_signal(a_lower) or _has_negation_signal(b_lower):
            return True
        # transition keyword in one fact + anchor keyword in the other
        a_has_anchor = any(kw in a_lower for kw in keywords)
        b_has_anchor = any(kw in b_lower for kw in keywords)
        a_has_transition = _has_transition_keyword(a_lower, category)
        b_has_transition = _has_transition_keyword(b_lower, category)
        if (a_has_anchor and b_has_transition) or (b_has_anchor and a_has_transition):
            return True

    return False


def compute_staleness_score(
    fact: MemoryFact,
    category: FactCategory,
    all_facts: list[MemoryFact],
    now: datetime,
    decay_rates: dict[FactCategory, float] | None = None,
) -> tuple[float, bool, str | None]:
    """Returns (score, has_contradiction, contradicted_by_id)."""
    reference_time = fact.last_confirmed_at or fact.created_at
    age_days = max((now - reference_time).days, 0)

    rates = decay_rates if decay_rates is not None else DECAY_RATES
    decay_rate = rates[category]
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
