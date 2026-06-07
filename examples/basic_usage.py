import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from memlint import StaleDetector
from memlint.adapters.json_adapter import load_from_json

facts = load_from_json(os.path.join(os.path.dirname(__file__), "sample_memories.json"))
detector = StaleDetector()
report = detector.check(facts)

print(f"Total: {report.total_facts} | Flagged: {len(report.flagged)}")
for result in report.flagged:
    print(f"  [{result.staleness_level.value.upper()}] {result.content}")
    print(f"    Reason: {result.reason}")
    print(f"    Action: {result.recommendation}")
