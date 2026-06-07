# Changelog

## 0.1.1 — 2026-06-07

- Added `__version__` exposed at package level
- Added `py.typed` marker for PEP 561 type checker support
- Added `classify_fact` to top-level exports
- Added PyPI keywords and classifiers for discoverability
- Added optional extras: `memlint[anthropic]`, `memlint[nvidia]`, `memlint[ollama]`, `memlint[bedrock]`
- Added docstrings to `StaleDetector`, `MemoryFact`, `DetectionReport`, and public methods
- Added GitHub Actions CI (Python 3.11, 3.12, 3.13)
- Removed build artifacts and internal tooling from git tracking

## 0.1.0 — 2026-06-07

- Initial release
- Staleness scoring with decay rates, confirmation bonus, contradiction detection
- Rule-based fact classifier with optional LLM classification
- Sync and async API: `check`, `check_one`, `filter_safe` and async variants
- Supports any LLM backend with `invoke()` / `ainvoke()` (OpenAI, Anthropic, NVIDIA NIM, Ollama, Bedrock)
- JSON and Mem0 adapters
- LangChain tool wrappers
- CLI with rich table output
