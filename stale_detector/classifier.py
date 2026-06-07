from stale_detector.models import FactCategory

CATEGORY_KEYWORDS: dict[FactCategory, list[str]] = {
    FactCategory.LOCATION: [
        "lives", "located", "based in", "address", "city", "country",
        "office", "moved to", "residing", "hometown", "location",
    ],
    FactCategory.EMPLOYMENT: [
        "works at", "employed", "job", "role", "position", "company",
        "organization", "joined", "hired", "manager", "team", "department",
        "title", "consultant", "engineer", "analyst", "intern",
    ],
    FactCategory.PROJECT: [
        "project", "building", "repo", "codebase", "app", "tool",
        "working on", "developing", "implementing", "stack", "framework",
        "library", "version", "api", "endpoint", "deployed", "launched",
    ],
    FactCategory.PREFERENCE: [
        "prefers", "likes", "favorite", "enjoys", "uses", "dislikes",
        "wants", "chooses", "opts for", "theme", "mode", "setting",
        "style", "approach",
    ],
    FactCategory.RELATIONSHIP: [
        "friend", "colleague", "manager", "reports to", "partner",
        "teammate", "mentor", "client", "collaborator", "family",
    ],
    FactCategory.IDENTITY: [
        "name is", "called", "age", "born", "nationality", "speaks",
        "gender", "education", "degree", "graduated", "alumni",
    ],
    FactCategory.EPISODIC: [
        "today", "yesterday", "last week", "this morning", "just",
        "recently", "earlier", "said that", "mentioned", "asked about",
        "discussed", "fixed", "resolved", "debugging",
    ],
    FactCategory.SYSTEM_FACT: [
        "python version", "node version", "npm", "pip", "docker",
        "os", "operating system", "machine", "cpu", "ram", "disk",
        "installed", "configured", "environment", "env", ".env",
    ],
}

CLASSIFY_PROMPT = """You are classifying a memory fact into exactly one category.

Categories:
- location: where someone lives, works, or is based
- employment: job, company, role, title, team
- project: software projects, tools being built, tech stack
- preference: likes, dislikes, habits, settings
- relationship: people the user knows or works with
- identity: name, age, education, nationality, languages spoken
- episodic: time-specific events, recent actions, things that happened
- system_fact: software versions, OS, environment config
- unknown: does not fit any category

Memory fact: "{fact}"

Respond with ONLY the category name, nothing else. Example: "employment"
"""


def _rule_based_classify(content: str) -> FactCategory:
    lower = content.lower()
    scores: dict[FactCategory, int] = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in lower)
        if hits > 0:
            scores[category] = hits
    if not scores:
        return FactCategory.UNKNOWN
    return max(scores, key=lambda c: scores[c])


def _llm_classify(content: str, llm_provider: str, model: str, llm=None) -> FactCategory:
    if llm is None:
        import os
        from langchain_openai import ChatOpenAI
        api_key = (
            os.getenv("OPENAI_API_KEY") if llm_provider == "openai"
            else os.getenv("ANTHROPIC_API_KEY")
        )
        if not api_key:
            raise ValueError(f"No API key found for provider {llm_provider!r}")
        llm = ChatOpenAI(model=model, temperature=0, api_key=api_key)

    prompt = CLASSIFY_PROMPT.format(fact=content)
    try:
        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content=prompt)]
    except ImportError:
        messages = [{"role": "user", "content": prompt}]

    response = llm.invoke(messages)
    raw = response.content.strip().lower()
    try:
        return FactCategory(raw)
    except ValueError:
        raise ValueError(f"LLM returned unrecognized category: {raw!r}")


def classify_fact(
    content: str,
    use_llm: bool = False,
    llm_provider: str = "openai",
    model: str = "gpt-4o-mini",
    llm=None,
) -> FactCategory:
    if use_llm:
        try:
            return _llm_classify(content, llm_provider, model, llm=llm)
        except Exception:
            pass
    return _rule_based_classify(content)


async def _async_llm_classify(content: str, llm_provider: str, model: str, llm=None) -> FactCategory:
    if llm is None:
        import os
        from langchain_openai import ChatOpenAI
        api_key = (
            os.getenv("OPENAI_API_KEY") if llm_provider == "openai"
            else os.getenv("ANTHROPIC_API_KEY")
        )
        if not api_key:
            raise ValueError(f"No API key found for provider {llm_provider!r}")
        llm = ChatOpenAI(model=model, temperature=0, api_key=api_key)

    prompt = CLASSIFY_PROMPT.format(fact=content)
    try:
        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content=prompt)]
    except ImportError:
        messages = [{"role": "user", "content": prompt}]

    response = await llm.ainvoke(messages)
    raw = response.content.strip().lower()
    try:
        return FactCategory(raw)
    except ValueError:
        raise ValueError(f"LLM returned unrecognized category: {raw!r}")


async def classify_fact_async(
    content: str,
    use_llm: bool = False,
    llm_provider: str = "openai",
    model: str = "gpt-4o-mini",
    llm=None,
) -> FactCategory:
    if use_llm:
        try:
            return await _async_llm_classify(content, llm_provider, model, llm=llm)
        except Exception:
            pass
    return _rule_based_classify(content)
