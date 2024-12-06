from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from data_agora import Agora, AgoraConfig
from data_agora.core.llms.test import TestLLM
from data_agora.core.parsers import InstanceGenerationParser, QualityEnhancementParser, ResponseGenerationParser
from data_agora.core.prompt_loaders import (
    InstanceGenerationPromptLoader,
    QualityEnhancementPromptLoader,
    ResponseGenerationPromptLoader,
)
from data_agora.core.validators import CodeValidator, GeneralValidator, MathValidator


@pytest.fixture
def seed_data_paths():
    """Fixture for seed data file paths"""
    return {
        "instance_generation": {
            "math": "./seed_data/math_seed_data.json",
            "code": "./seed_data/code_seed_data.json",
            "general": "./seed_data/general_seed_data.json",
        },
        "response_generation": {
            "math": "./seed_data/magpie_math_seed_data_10000.json",
            "code": "./seed_data/magpie_code_seed_data_10000.json",
            "general": "./seed_data/magpie_general_seed_data_10000.json",
        },
        "quality_enhancement": {
            "math": "./seed_data/webinstruct_math_seed_data_10000.json",
            "general": "./seed_data/webinstruct_general_seed_data_10000.json",
        },
    }


@pytest.fixture
def prompt_paths():
    """Fixture for prompt template file paths"""
    return {
        "instance_generation": {
            "math": "./_prompts/math_prompt.txt",
            "code": "./_prompts/code_prompt.txt",
            "general": "./_prompts/general_prompt.txt",
        },
        "response_generation": {
            "math": "./_prompts/magpie_math_prompt.txt",
            "code": "./_prompts/magpie_code_prompt.txt",
            "general": "./_prompts/magpie_general_prompt.txt",
        },
        "quality_enhancement": {
            "math": "./_prompts/webinstruct_math_prompt.txt",
            "general": "./_prompts/webinstruct_general_prompt.txt",
        },
    }


@pytest.fixture
def mock_seed_data():
    """Fixture for mock seed data"""
    return [
        {"instruction": "test instruction 1", "input": "test input 1", "response": "test output 1"},
        {"instruction": "test instruction 2", "input": "test input 2", "response": "test output 2"},
        {"instruction": "test instruction 3", "input": "test input 3", "response": "test output 3"},
    ]


@pytest.fixture
def mock_prompt():
    """Fixture for mock prompt template with multiple example placeholders"""
    return """Example 1:
Input: <input1>
Output: <output1>

Example 2:
Input: <input2>
Output: <output2>

Example 3:
Input: <input3>
Output: <output3>

Now generate a new example:
Input: <input@>
Output:"""


@pytest.fixture
def placeholder_formats():
    """Fixture for placeholder format definitions"""
    return {
        "demonstration_input_placeholder": "<input{}>",
        "demonstration_output_placeholder": "<output{}>",
        "test_input_placeholder": "<input@>",
        "test_output_placeholder": "<output@>",
        "input_trigger": "INPUT:",
        "output_trigger": "OUTPUT:",
        "stop_phrase": "[END]",
    }


@pytest.fixture
def llm():
    """Fixture for test LLM instance"""
    return TestLLM("test")


def test_instance_generation(llm, mock_seed_data, mock_prompt, placeholder_formats):
    """Test the instance generation workflow"""
    prompt_loader = InstanceGenerationPromptLoader(
        prompt_template=mock_prompt, seed_data=mock_seed_data, num_fewshot=3, placeholder_formats=placeholder_formats
    )

    parser = InstanceGenerationParser()
    validator = GeneralValidator(tokenizer_name="meta-llama/Meta-Llama-3.1-8B-Instruct", max_tokens=4096)

    alchemy = Agora(
        # generation_type="sampling",
        llm=llm,
        placeholder_formats=placeholder_formats,
        prompt_loader=prompt_loader,
        parser=parser,
        validator=validator,
    )

    result = alchemy.run_single()

    assert result is not None
    assert "instruction" in result
    assert "response" in result
    assert "metadata" in result


def test_response_generation(llm, mock_seed_data, mock_prompt, placeholder_formats):
    """Test the response generation workflow"""
    prompt_loader = ResponseGenerationPromptLoader(
        prompt_template=mock_prompt, seed_data=mock_seed_data, placeholder_formats=placeholder_formats
    )

    parser = ResponseGenerationParser()
    validator = GeneralValidator(tokenizer_name="meta-llama/Meta-Llama-3.1-8B-Instruct", max_tokens=4096)

    alchemy = Agora(
        # generation_type="sampling",
        llm=llm,
        placeholder_formats=placeholder_formats,
        prompt_loader=prompt_loader,
        parser=parser,
        validator=validator,
    )

    result = alchemy.run_single()

    assert result is not None
    assert "instruction" in result
    assert "response" in result
    assert "metadata" in result


@pytest.mark.parametrize("domain", ["math", "code", "general"])
def test_different_domains(llm, mock_seed_data, mock_prompt, placeholder_formats, domain):
    """Test generation with different domain validators"""
    # Select validator based on domain
    validator_map = {"math": MathValidator, "code": CodeValidator, "general": GeneralValidator}

    validator_cls = validator_map[domain]
    validator = validator_cls(tokenizer_name="meta-llama/Meta-Llama-3.1-8B-Instruct", max_tokens=4096)

    # Initialize components
    prompt_loader = InstanceGenerationPromptLoader(
        prompt_template=mock_prompt, seed_data=mock_seed_data, num_fewshot=3, placeholder_formats=placeholder_formats
    )
    parser = InstanceGenerationParser()

    # Create and configure Agora instance
    alchemy = Agora(
        # generation_type="sampling",
        llm=llm,
        placeholder_formats=placeholder_formats,
        prompt_loader=prompt_loader,
        parser=parser,
        validator=validator,
        sampling_params={"domain": domain},  # ONLY FOR TESTING
    )

    # Run single generation
    MAX_RETRIES = 20
    trial = 0

    while trial < MAX_RETRIES:
        result = alchemy.run_single()
        trial += 1
        if result is not None:
            break

    # Assertions
    assert result is not None
    assert "instruction" in result
    assert "response" in result
    assert "metadata" in result


def test_parallel_generation(llm, mock_seed_data, mock_prompt, placeholder_formats):
    """Test parallel generation with multiple threads"""
    prompt_loader = InstanceGenerationPromptLoader(
        prompt_template=mock_prompt, seed_data=mock_seed_data, num_fewshot=3, placeholder_formats=placeholder_formats
    )
    parser = InstanceGenerationParser()
    validator = GeneralValidator(tokenizer_name="meta-llama/Meta-Llama-3.1-8B-Instruct", max_tokens=4096)

    alchemy = Agora(
        # generation_type="sampling",
        llm=llm,
        placeholder_formats=placeholder_formats,
        prompt_loader=prompt_loader,
        parser=parser,
        validator=validator,
    )

    num_instances = 10
    results = alchemy.run(num_instances=num_instances, num_threads=4)

    assert len(results) == num_instances
    for result in results:
        assert "instruction" in result
        assert "response" in result
        assert "metadata" in result


# def test_error_handling(llm, mock_seed_data, mock_prompt, placeholder_formats):
#     """Test error handling and retry logic"""
#     # Create a mock LLM that fails occasionally
#     failing_llm = Mock()
#     failing_llm.chat.side_effect = [
#         Exception("API Error"),  # First call fails
#         {"choices": [{"message": {"content": "Success"}, "finish_reason": "stop"}]},  # Second call succeeds
#     ] * 1000  # Repeat the pattern 1000 times


#     prompt_loader = InstanceGenerationPromptLoader(
#         prompt_template=mock_prompt, seed_data=mock_seed_data, num_fewshot=3, placeholder_formats=placeholder_formats
#     )
#     parser = InstanceGenerationParser()
#     validator = GeneralValidator(tokenizer_name="meta-llama/Meta-Llama-3.1-8B-Instruct", max_tokens=4096)

#     config = AgoraConfig(max_retries=100, retry_delay=0.01)
#     alchemy = Agora(
#         generation_type="sampling",
#         llm=failing_llm,
#         placeholder_formats=placeholder_formats,
#         prompt_loader=prompt_loader,
#         parser=parser,
#         validator=validator,
#         config=config,
#     )

#     # Should eventually succeed after retry
#     result = alchemy.run(num_instances=10)
#     assert result is not None


if __name__ == "__main__":
    pytest.main([__file__])
