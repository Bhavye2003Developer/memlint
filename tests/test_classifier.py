from stale_detector.classifier import classify_fact
from stale_detector.models import FactCategory


def test_classifies_location():
    assert classify_fact("User lives in Delhi") == FactCategory.LOCATION


def test_classifies_employment():
    assert classify_fact("User works at PwC as a consultant") == FactCategory.EMPLOYMENT


def test_classifies_project():
    assert classify_fact("Building pract-agents using LangGraph") == FactCategory.PROJECT


def test_classifies_preference():
    assert classify_fact("User prefers Python over JavaScript") == FactCategory.PREFERENCE


def test_unknown_fallback():
    assert classify_fact("The sky is blue") == FactCategory.UNKNOWN


def test_case_insensitive():
    assert classify_fact("USER LIVES IN DELHI") == FactCategory.LOCATION


def test_most_hits_wins():
    # EMPLOYMENT has: "works at", "title", "analyst" = 3 hits; PREFERENCE has: "prefers" = 1 hit
    assert classify_fact("User works at company, title is analyst, prefers Python") == FactCategory.EMPLOYMENT
