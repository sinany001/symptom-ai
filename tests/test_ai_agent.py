from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ai_agent import (
    SymptomInformation,
    extract_symptom_information,
)


def test_extracts_structured_information():
    expected = SymptomInformation(
        symptoms=["cough", "fever"],
        duration="2 days",
        severity="moderate",
        missing_information=["Other associated symptoms"],
        follow_up_questions=["Do you have difficulty breathing?"],
        uncertainty_notes=[],
    )

    fake_response = SimpleNamespace(
    message=SimpleNamespace(
        content=expected.model_dump_json()
    )
)
    

    with patch("ai_agent.ollama.chat", return_value=fake_response):
        result = extract_symptom_information(
            "I have a cough and fever for 2 days."
        )

    assert result.symptoms == ["cough", "fever"]
    assert result.duration == "2 days"
    assert result.severity == "moderate"


def test_rejects_too_short_input():
    with pytest.raises(ValueError):
        extract_symptom_information("hi")


def test_handles_missing_parsed_result():
    from types import SimpleNamespace

    fake_response = SimpleNamespace(
        message=SimpleNamespace(content="")
    )

    with patch("ai_agent.ollama.chat", return_value=fake_response):
        with pytest.raises(RuntimeError):
            extract_symptom_information("I have a cough.")
    

    with patch("ai_agent.ollama.chat", return_value=fake_response):
        with pytest.raises(RuntimeError):
            extract_symptom_information("I have a cough.")
