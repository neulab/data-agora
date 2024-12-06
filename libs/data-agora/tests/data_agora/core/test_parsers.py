from typing import Dict

import pytest

from data_agora.core.parsers import (
    InstanceGenerationDictParser,
    InstanceGenerationParser,
    QualityEnhancementDictParser,
    QualityEnhancementParser,
    ResponseGenerationDictParser,
    ResponseGenerationParser,
)


# Test fixtures
@pytest.fixture
def placeholder_formats() -> Dict[str, str]:
    return {"input_trigger": "Instruction:", "output_trigger": "Response:", "stop_phrase": "<END>"}


@pytest.fixture
def sample_dict_output() -> Dict[str, str]:
    return {
        "instruction": "Write a poem about space",
        "response": "Stars twinkle in the night\nPlanets dance with delight",
    }


class TestInstanceGenerationParser:
    def test_basic_parsing(self, placeholder_formats):
        parser = InstanceGenerationParser()
        parsed_output = """
        Instruction: Write a poem about space\nResponse: Stars twinkle in the night\nPlanets dance with delight\n<END>
        """
        result = parser.parse("", parsed_output, placeholder_formats)

        assert result["instruction"] == "Write a poem about space"
        assert result["response"] == "Stars twinkle in the night\nPlanets dance with delight"

    def test_multiple_delimiters(self, placeholder_formats):
        parser = InstanceGenerationParser()
        parsed_output = """
        Instruction: First instruction\nSome text\nInstruction: Write a poem about space\nResponse: Stars twinkle in the night\n<END>
        """
        result = parser.parse("", parsed_output, placeholder_formats)

        assert result["instruction"] == "Write a poem about space"
        assert result["response"] == "Stars twinkle in the night"

    def test_missing_stop_phrase(self, placeholder_formats):
        parser = InstanceGenerationParser()
        parsed_output = """
        Instruction: Write a poem\nResponse: A simple poem
        """
        result = parser.parse("", parsed_output, placeholder_formats)

        assert result["instruction"] == "Write a poem"
        assert result["response"] == "A simple poem"


class TestResponseGenerationParser:
    def test_basic_parsing(self, placeholder_formats):
        parser = ResponseGenerationParser()
        prompt = "Instruction: Write a poem about space"
        parsed_output = "Response: Stars twinkle in the night<END>"

        result = parser.parse(prompt, parsed_output, placeholder_formats)

        assert result["instruction"] == "Write a poem about space"
        assert result["response"] == "Stars twinkle in the night"

    def test_multiline_response(self, placeholder_formats):
        parser = ResponseGenerationParser()
        prompt = "Instruction: Write a haiku"
        parsed_output = """Response: Autumn leaves falling
        Gentle breeze whispers softly
        Nature's lullaby
        <END>"""

        result = parser.parse(prompt, parsed_output, placeholder_formats)

        assert result["instruction"] == "Write a haiku"
        assert "Autumn leaves falling" in result["response"]
        assert "Nature's lullaby" in result["response"]

    def test_with_extra_content(self, placeholder_formats):
        parser = ResponseGenerationParser()
        prompt = "Some prefix text\nInstruction: Write a poem"
        parsed_output = "Some prefix\nResponse: A simple poem\nSome suffix"

        result = parser.parse(prompt, parsed_output, placeholder_formats)

        assert result["instruction"] == "Write a poem"
        assert result["response"] == "A simple poem\nSome suffix"


class TestQualityEnhancementParser:
    def test_basic_parsing(self, placeholder_formats):
        parser = QualityEnhancementParser()
        prompt = """
        Instruction: Improve this text
        Response: Original response
        """
        parsed_output = """
        Instruction: This is an improved text
        Response: Enhanced response<END>"""

        result = parser.parse(prompt, parsed_output, placeholder_formats)

        assert result["instruction"] == "This is an improved text"
        assert result["response"] == "Enhanced response"

    def test_complex_input(self, placeholder_formats):
        parser = QualityEnhancementParser()
        prompt = """
        Context: Some context
        Instruction: Improve this text with multiple
        lines and special chars: !@#$%
        Response: Original text
        """
        parsed_output = """
        Instruction: This is an improved text
        Response: Enhanced text with
        multiple lines and
        special chars: !@#$%
        <END>
        """

        result = parser.parse(prompt, parsed_output, placeholder_formats)

        assert "This is an improved text" in result["instruction"]
        assert "Enhanced text" in result["response"]


class TestInstanceGenerationDictParser:
    def test_basic_dict_parsing(self, sample_dict_output):
        parser = InstanceGenerationDictParser()
        result = parser.parse("", sample_dict_output)

        assert result == sample_dict_output
        assert result["instruction"] == "Write a poem about space"
        assert "Stars twinkle" in result["response"]

    def test_extra_fields(self):
        parser = InstanceGenerationDictParser()
        output_with_extra = {"instruction": "Task", "response": "Answer", "extra_field": "Should be ignored"}

        result = parser.parse("", output_with_extra)

        assert "instruction" in result
        assert "response" in result
        assert len(result) == 2


class TestResponseGenerationDictParser:
    def test_basic_dict_parsing(self, sample_dict_output):
        parser = ResponseGenerationDictParser()
        result = parser.parse("", sample_dict_output)

        assert "instruction" in result
        assert "response" in result
        assert result["instruction"] == "Write a poem about space"
        assert "Stars twinkle" in result["response"]


class TestQualityEnhancementDictParser:
    def test_basic_dict_parsing(self, sample_dict_output):
        parser = QualityEnhancementDictParser()
        result = parser.parse("", sample_dict_output)

        assert result == sample_dict_output
        assert "instruction" in result
        assert "response" in result

    def test_missing_fields(self):
        parser = QualityEnhancementDictParser()
        incomplete_output = {
            "instruction": "Task"
            # missing response field
        }

        with pytest.raises(KeyError):
            parser.parse("", incomplete_output)
