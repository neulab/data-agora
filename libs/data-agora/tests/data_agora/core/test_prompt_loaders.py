import random
from typing import Dict, List

import pytest

from data_agora.core.prompt_loaders import (
    InstanceGenerationPromptLoader,
    PromptResult,
    QualityEnhancementPromptLoader,
    ResponseGenerationPromptLoader,
    load_prompt_loader,
)


# Test data fixtures
@pytest.fixture
def sample_seed_data() -> List[Dict[str, str]]:
    return [
        {"instruction": "Task 1", "response": "Response 1"},
        {"instruction": "Task 2", "response": "Response 2"},
        {"instruction": "Task 3", "response": "Response 3"},
        {"instruction": "Task 4", "response": "Response 4"},
    ]


@pytest.fixture
def sample_template() -> str:
    return "Examples:\n<input1>\n<output1>\n<input2>\n<output2>\nNew task:"


@pytest.fixture
def custom_placeholder_formats() -> Dict[str, str]:
    return {
        "demonstration_input_placeholder": "[[input@]]",
        "demonstration_output_placeholder": "[[output@]]",
    }


# Instance Generation Tests
class TestInstanceGenerationPromptLoader:
    def test_initialization_valid(self, sample_seed_data, sample_template):
        loader = InstanceGenerationPromptLoader(
            prompt_template=sample_template, seed_data=sample_seed_data, num_fewshot=2
        )
        assert loader.num_fewshot == 2
        assert loader.seed_data == sample_seed_data
        assert loader.prompt_template == sample_template

    def test_initialization_invalid_num_fewshot(self, sample_seed_data, sample_template):
        with pytest.raises(ValueError, match="num_fewshot must be greater than 0"):
            InstanceGenerationPromptLoader(prompt_template=sample_template, seed_data=sample_seed_data, num_fewshot=0)

    def test_initialization_insufficient_seed_data(self, sample_seed_data, sample_template):
        with pytest.raises(ValueError, match="Not enough seed data"):
            InstanceGenerationPromptLoader(prompt_template=sample_template, seed_data=sample_seed_data, num_fewshot=5)

    def test_custom_placeholder_formats(self, sample_seed_data):
        template = "Examples:\n[[input1]]\n[[output1]]\n[[input2]]\n[[output2]]\nNew task:"
        custom_formats = {
            "demonstration_input_placeholder": "[[input@]]",
            "demonstration_output_placeholder": "[[output@]]",
        }
        loader = InstanceGenerationPromptLoader(
            prompt_template=template, seed_data=sample_seed_data, num_fewshot=2, placeholder_formats=custom_formats
        )
        result = loader.prepare()
        assert isinstance(result, PromptResult)
        assert "[[input" not in result.prompt
        assert "[[output" not in result.prompt

    def test_prepare_output(self, sample_seed_data, sample_template):
        random.seed(42)  # For reproducible testing
        loader = InstanceGenerationPromptLoader(
            prompt_template=sample_template, seed_data=sample_seed_data, num_fewshot=2
        )
        result = loader.prepare()
        assert isinstance(result, PromptResult)
        assert isinstance(result.metadata, dict)
        assert result.metadata["num_examples"] == 2
        assert len(result.metadata["examples"]) == 2


# Response Generation Tests
class TestResponseGenerationPromptLoader:
    def test_initialization_valid(self, sample_seed_data):
        template = "Generate a response for: <input>"
        loader = ResponseGenerationPromptLoader(prompt_template=template, seed_data=sample_seed_data)
        assert loader.prompt_template == template
        assert loader.seed_data == sample_seed_data
        assert loader.current_index == 0

    def test_missing_placeholder(self, sample_seed_data):
        template = "Generate a response for: "  # Missing placeholder
        with pytest.raises(ValueError, match="missing required placeholder"):
            ResponseGenerationPromptLoader(prompt_template=template, seed_data=sample_seed_data)

    def test_prepare_sequential_processing(self, sample_seed_data):
        template = "Generate a response for: <input>"
        loader = ResponseGenerationPromptLoader(prompt_template=template, seed_data=sample_seed_data)

        # First preparation
        result1 = loader.prepare()
        assert result1.metadata["index"] == 0
        assert result1.metadata["instruction"] == sample_seed_data[0]["instruction"]

        # Second preparation
        result2 = loader.prepare()
        assert result2.metadata["index"] == 1
        assert result2.metadata["instruction"] == sample_seed_data[1]["instruction"]

    def test_preparation_exhaustion(self, sample_seed_data):
        template = "Generate a response for: <input>"
        loader = ResponseGenerationPromptLoader(prompt_template=template, seed_data=sample_seed_data)

        # Exhaust all data
        for _ in range(len(sample_seed_data)):
            loader.prepare()

        with pytest.raises(StopIteration):
            loader.prepare()


# Quality Enhancement Tests
class TestQualityEnhancementPromptLoader:
    def test_initialization_valid(self, sample_seed_data):
        template = "Instruction: <input>\nResponse: <output>\nEnhance the response."
        loader = QualityEnhancementPromptLoader(prompt_template=template, seed_data=sample_seed_data)
        assert loader.prompt_template == template
        assert loader.seed_data == sample_seed_data
        assert loader.current_index == 0

    def test_missing_placeholders(self, sample_seed_data):
        template = "Enhance this response:"  # Missing both placeholders
        with pytest.raises(ValueError, match="missing required placeholders"):
            QualityEnhancementPromptLoader(prompt_template=template, seed_data=sample_seed_data)

    def test_prepare_output_format(self, sample_seed_data):
        template = "Instruction: <input>\nResponse: <output>\nEnhance the response."
        loader = QualityEnhancementPromptLoader(prompt_template=template, seed_data=sample_seed_data)
        result = loader.prepare()
        assert isinstance(result, PromptResult)
        assert result.metadata["index"] == 0
        assert result.metadata["instruction"] == sample_seed_data[0]["instruction"]
        assert result.metadata["response"] == sample_seed_data[0]["response"]
        assert "<input>" not in result.prompt
        assert "<output>" not in result.prompt

    def test_invalid_seed_data_format(self):
        template = "Instruction: <input>\nResponse: <output>\nEnhance the response."
        invalid_seed_data = [{"instruction": "Task 1"}]  # Missing response
        loader = QualityEnhancementPromptLoader(prompt_template=template, seed_data=invalid_seed_data)
        with pytest.raises(ValueError, match="missing required keys"):
            loader.prepare()


# Factory Function Tests
class TestLoadPromptLoader:
    def test_valid_loader_types(self, sample_seed_data, sample_template):
        # Test instance generation loader
        loader1 = load_prompt_loader(
            "instance_generation", prompt_template=sample_template, seed_data=sample_seed_data, num_fewshot=2
        )
        assert isinstance(loader1, InstanceGenerationPromptLoader)

        # Test response generation loader
        loader2 = load_prompt_loader(
            "response_generation", prompt_template="Generate a response for: <input>", seed_data=sample_seed_data
        )
        assert isinstance(loader2, ResponseGenerationPromptLoader)

        # Test quality enhancement loader
        loader3 = load_prompt_loader(
            "quality_enhancement",
            prompt_template="Instruction: <input>\nResponse: <output>\nEnhance the response.",
            seed_data=sample_seed_data,
        )
        assert isinstance(loader3, QualityEnhancementPromptLoader)

    def test_invalid_loader_type(self, sample_seed_data, sample_template):
        with pytest.raises(ValueError, match="Invalid prompt loader type"):
            load_prompt_loader("invalid_type", prompt_template=sample_template, seed_data=sample_seed_data)
