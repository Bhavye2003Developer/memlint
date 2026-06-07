from datetime import datetime
from stale_detector.models import (
    FactCategory, MemoryFact, StalenessResult, DetectionReport, StalenessLevel,
)
from stale_detector.classifier import classify_fact
from stale_detector.scorer import (
    compute_staleness_score, determine_level, build_reason, build_recommendation,
)


class StaleDetector:
    def __init__(
        self,
        use_llm: bool = False,
        llm_provider: str = "openai",
        model: str = "gpt-4o-mini",
    ):
        self._use_llm = use_llm
        self._llm_provider = llm_provider
        self._model = model

    def _classify(self, fact: MemoryFact) -> FactCategory:
        if fact.category is not None:
            return fact.category
        return classify_fact(
            fact.content,
            use_llm=self._use_llm,
            llm_provider=self._llm_provider,
            model=self._model,
        )

    def check_one(
        self,
        fact: MemoryFact,
        context_facts: list[MemoryFact] | None = None,
        now: datetime | None = None,
    ) -> StalenessResult:
        if now is None:
            now = datetime.utcnow()
        all_facts = context_facts if context_facts is not None else [fact]
        category = self._classify(fact)
        score, has_contradiction, contradicted_by = compute_staleness_score(
            fact, category, all_facts, now
        )
        level = determine_level(score)
        reference_time = fact.last_confirmed_at or fact.created_at
        age_days = max((now - reference_time).days, 0)
        return StalenessResult(
            fact_id=fact.id,
            content=fact.content,
            category=category,
            staleness_score=round(score, 4),
            staleness_level=level,
            age_days=age_days,
            reason=build_reason(age_days, category, fact.confirmation_count, has_contradiction, score),
            recommendation=build_recommendation(level),
            has_contradiction=has_contradiction,
            contradicted_by=contradicted_by,
        )

    def check(
        self,
        facts: list[MemoryFact],
        now: datetime | None = None,
    ) -> DetectionReport:
        if now is None:
            now = datetime.utcnow()
        results = [self.check_one(f, context_facts=facts, now=now) for f in facts]
        counts: dict[StalenessLevel, int] = {level: 0 for level in StalenessLevel}
        for r in results:
            counts[r.staleness_level] += 1
        return DetectionReport(
            checked_at=now,
            total_facts=len(facts),
            fresh_count=counts[StalenessLevel.FRESH],
            aging_count=counts[StalenessLevel.AGING],
            stale_count=counts[StalenessLevel.STALE],
            expired_count=counts[StalenessLevel.EXPIRED],
            results=results,
        )

    def filter_safe(
        self,
        facts: list[MemoryFact],
        now: datetime | None = None,
    ) -> list[MemoryFact]:
        report = self.check(facts, now)
        safe_ids = {r.fact_id for r in report.safe}
        return [f for f in facts if f.id in safe_ids]
