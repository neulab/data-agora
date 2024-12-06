from typing import Dict, List

import pytest
from transformers import AutoTokenizer

from data_agora.core.validators import (
    CodeValidator,
    GeneralValidator,
    MathValidator,
    validate_dict_format,
    validate_forbidden_keywords,
    validate_keywords,
    validate_length,
)


# Mock tokenizer for testing
class MockTokenizer:
    def encode(self, text):
        # Simple mock that returns 1 token per word
        return text.split()


@pytest.fixture
def mock_tokenizer(monkeypatch):
    def mock_from_pretrained(*args, **kwargs):
        return MockTokenizer()

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", mock_from_pretrained)
    return MockTokenizer()


@pytest.fixture
def sample_placeholder_formats():
    return {"input_trigger": "<input>", "output_trigger": "<output>", "stop_phrase": "<end>"}


class TestGeneralValidator:
    def test_basic_validation(self, mock_tokenizer):
        validator = GeneralValidator("dummy-tokenizer", max_tokens=100)
        assert validator.validate("Short instruction", "Short response") == True

    def test_length_validation(self, mock_tokenizer):
        validator = GeneralValidator("dummy-tokenizer", max_tokens=5)
        assert validator.validate("Short instruction", "This response is too long to be valid") == False

    def test_placeholder_validation(self, mock_tokenizer, sample_placeholder_formats):
        validator = GeneralValidator("dummy-tokenizer", max_tokens=100, placeholder_formats=sample_placeholder_formats)
        # Response contains forbidden placeholder
        assert validator.validate("Instruction", "Response with <input> placeholder") == False

        # Valid response without placeholders
        assert validator.validate("Instruction", "Valid response") == True


class TestMathValidator:
    def test_basic_validation(self, mock_tokenizer):
        validator = MathValidator("dummy-tokenizer", max_tokens=100)
        valid_response = "Here's the solution: [RESULT]42[/RESULT]"
        assert validator.validate("Math problem", valid_response) == True

    def test_missing_result_tags(self, mock_tokenizer):
        validator = MathValidator("dummy-tokenizer", max_tokens=100)
        invalid_response = "Here's the solution: 42"
        assert validator.validate("Math problem", invalid_response) == False

    def test_multiple_result_tags(self, mock_tokenizer):
        validator = MathValidator("dummy-tokenizer", max_tokens=100)
        invalid_response = "[RESULT]42[/RESULT][RESULT]43[/RESULT]"
        assert validator.validate("Math problem", invalid_response) == False

    def test_with_placeholders(self, mock_tokenizer, sample_placeholder_formats):
        validator = MathValidator("dummy-tokenizer", max_tokens=100, placeholder_formats=sample_placeholder_formats)
        invalid_response = "[RESULT]Solution with <input>[/RESULT]"
        assert validator.validate("Math problem", invalid_response) == False


class TestCodeValidator:
    def test_basic_validation(self, mock_tokenizer):
        validator = CodeValidator("dummy-tokenizer", max_tokens=100)
        valid_response = "```python\ndef add(a, b):\n    return a + b\n```"
        assert validator.validate("Write a function", valid_response) == True

    def test_unmatched_code_blocks(self, mock_tokenizer):
        validator = CodeValidator("dummy-tokenizer", max_tokens=100)
        invalid_response = "```python\ndef add(a, b):\n    return a + b"
        assert validator.validate("Write a function", invalid_response) == False

    def test_forbidden_keywords(self, mock_tokenizer):
        validator = CodeValidator("dummy-tokenizer", max_tokens=100)
        invalid_response = "```python\n# Here's an example usage\ndef add(a, b):\n    return a + b\n```"
        assert validator.validate("Write a function", invalid_response) == False

    def test_with_placeholders(self, mock_tokenizer, sample_placeholder_formats):
        validator = CodeValidator("dummy-tokenizer", max_tokens=100, placeholder_formats=sample_placeholder_formats)
        invalid_response = "```python\n# <input>\ndef add(a, b):\n    return a + b\n```"
        assert validator.validate("Write a function", invalid_response) == False


class TestValidateKeywords:
    def test_keyword_occurrence(self):
        text = "Hello world! Hello again!"
        assert validate_keywords(text, keyword_occurance={"Hello": 2}) == True
        assert validate_keywords(text, keyword_occurance={"Hello": 1}) == False

    def test_start_keywords(self):
        text = "Hello world!"
        assert validate_keywords(text, start_keywords=["Hello"]) == True
        assert validate_keywords(text, start_keywords=["World"]) == False

    def test_end_keywords(self):
        text = "Hello world!"
        assert validate_keywords(text, end_keywords=["world!"]) == True
        assert validate_keywords(text, end_keywords=["Hello"]) == False

    def test_multiple_conditions(self):
        text = "Hello world! Hello again!"
        assert (
            validate_keywords(text, keyword_occurance={"Hello": 2}, start_keywords=["Hello"], end_keywords=["again!"])
            == True
        )


class TestValidateLength:
    def test_within_limit(self, mock_tokenizer):
        assert validate_length("Short input", "Short output", mock_tokenizer, max_tokens=100) == True

    def test_exceeds_limit(self, mock_tokenizer):
        long_input = "Very long input " * 10
        long_output = "Very long output " * 10
        assert validate_length(long_input, long_output, mock_tokenizer, max_tokens=10) == False


class TestValidateForbiddenKeywords:
    def test_no_forbidden_keywords(self):
        text = "This is a valid response"
        forbidden = ["invalid", "error"]
        assert validate_forbidden_keywords(text, forbidden) == True

    def test_contains_forbidden_keywords(self):
        text = "This is an invalid response"
        forbidden = ["invalid", "error"]
        assert validate_forbidden_keywords(text, forbidden) == False

    def test_case_insensitive(self):
        text = "This is an INVALID response"
        forbidden = ["invalid", "error"]
        assert validate_forbidden_keywords(text, forbidden) == False


class TestValidateDictFormat:
    def test_valid_format(self):
        valid_output = "INPUT: Test instruction\nOUTPUT: Test response"
        assert validate_dict_format(valid_output, ["instruction", "response"]) == True

    def test_missing_input(self):
        invalid_output = "OUTPUT: Test response"
        assert validate_dict_format(invalid_output, ["instruction", "response"]) == False

    def test_missing_output(self):
        invalid_output = "INPUT: Test instruction"
        assert validate_dict_format(invalid_output, ["instruction", "response"]) == False

    def test_invalid_format(self):
        invalid_output = "Not in the correct format"
        assert validate_dict_format(invalid_output, ["instruction", "response"]) == False

    def test_extra_keys(self):
        valid_output = "INPUT: Test instruction\nOUTPUT: Test response"
        assert validate_dict_format(valid_output, ["instruction", "response", "extra"]) == False


# Integration tests
class TestValidatorIntegration:
    def test_general_validator_with_code(self, mock_tokenizer):
        validator = GeneralValidator("dummy-tokenizer", max_tokens=100)
        code_response = "```python\ndef add(a, b):\n    return a + b\n```"
        assert validator.validate("Write a function", code_response) == True

    def test_math_validator_with_code(self, mock_tokenizer):
        validator = MathValidator("dummy-tokenizer", max_tokens=100)
        code_response = "[RESULT]```python\ndef add(a, b):\n    return a + b\n```[/RESULT]"
        assert validator.validate("Write a function", code_response) == True

    def test_code_validator_with_math(self, mock_tokenizer):
        validator = CodeValidator("dummy-tokenizer", max_tokens=100)
        math_response = "```python\n[RESULT]42[/RESULT]\n```"
        assert validator.validate("Solve this", math_response) == True
