import json
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class AugmentationScenario(Enum):
    INSTANCE_GENERATION = "instance_generation"
    RESPONSE_GENERATION = "response_generation"
    QUALITY_ENHANCEMENT = "quality_enhancement"


@dataclass
class PromptResult:
    """Output from prompt preparation"""

    prompt: str
    metadata: Dict[str, Any]


class BasePromptLoader(ABC):
    """Abstract base class for prompt preparation with minimal requirements"""

    @abstractmethod
    def prepare(self) -> PromptResult:
        """Prepare a prompt based on scenario"""
        pass


class InstanceGenerationPromptLoader(BasePromptLoader):
    """Prompt loader for instance generation scenario"""

    def __init__(
        self,
        prompt_template: str,
        seed_data: List[Dict],
        num_fewshot: int,
        placeholder_formats: Dict[str, str] = None,
        num_sample_from_seed_data: Optional[int] = None,
    ):
        """Initialize the instance generation prompt loader.

        Args:
            prompt_template: Template string for prompt generation with <input1>, <output1>, etc.
            seed_data: List of examples to use for few-shot prompting
            num_fewshot: Number of few-shot examples to include (must be > 0)
            placeholder_formats: Optional dictionary with 'demonstration_input_placeholder' and 'demonstration_output_placeholder' keys defining custom placeholder formats.
                               The '@' symbol in the format will be replaced with the example number.
                               Default format uses '<input@>' and '<output@>'.
        """
        if num_fewshot <= 0:
            raise ValueError("num_fewshot must be greater than 0")
        if len(seed_data) < num_fewshot:
            raise ValueError(
                f"Not enough seed data ({len(seed_data)}) for requested number of few-shot examples ({num_fewshot})"
            )

        if num_sample_from_seed_data is None:
            num_sample_from_seed_data = num_fewshot
        if num_sample_from_seed_data > len(seed_data):
            raise ValueError(
                f"Number of samples from seed data ({num_sample_from_seed_data}) must be less than or equal to the length of seed data ({len(seed_data)})"
            )
        if num_sample_from_seed_data > num_fewshot:
            raise ValueError(
                f"Number of samples from seed data ({num_sample_from_seed_data}) must be less than or equal to the number of few-shot examples ({num_fewshot})"
            )

        self.num_sample_from_seed_data = num_sample_from_seed_data

        self.placeholder_formats = placeholder_formats or {
            "demonstration_input_placeholder": "<input@>",
            "demonstration_output_placeholder": "<output@>",
        }

        if not all(
            key in self.placeholder_formats
            for key in ["demonstration_input_placeholder", "demonstration_output_placeholder"]
        ):
            raise ValueError(
                "placeholder_formats must contain both 'demonstration_input_placeholder' and 'demonstration_output_placeholder' keys"
            )

        self._validate_placeholders(prompt_template, num_fewshot)

        self.prompt_template = prompt_template
        self._init_seed_data = seed_data
        self._init_seed_data_len = len(seed_data)
        self.seed_data = seed_data
        self.num_fewshot = num_fewshot

    def _validate_placeholders(self, prompt_template: str, num_fewshot: int):
        # Validate template contains required placeholders
        expected_placeholders = set()
        for i in range(1, num_fewshot + 1):
            # Check for both @ style and numbered style placeholders
            input_placeholder_at = self.placeholder_formats["demonstration_input_placeholder"].replace("@", str(i))
            output_placeholder_at = self.placeholder_formats["demonstration_output_placeholder"].replace("@", str(i))
            input_placeholder_num = f"<input{i}>"
            output_placeholder_num = f"<output{i}>"
            expected_placeholders.update(
                [input_placeholder_at, output_placeholder_at, input_placeholder_num, output_placeholder_num]
            )

        # Create regex patterns for both styles
        input_pattern_at = re.escape(self.placeholder_formats["demonstration_input_placeholder"]).replace("@", r"\d+")
        output_pattern_at = re.escape(self.placeholder_formats["demonstration_output_placeholder"]).replace(
            "@", r"\d+"
        )
        input_pattern_num = r"<input\d+>"
        output_pattern_num = r"<output\d+>"
        pattern = f"({input_pattern_at}|{output_pattern_at}|{input_pattern_num}|{output_pattern_num})"

        found_placeholders = set(re.findall(pattern, prompt_template))

        # Check if any valid placeholder format exists
        valid_placeholders = set()
        for placeholder in found_placeholders:
            # Extract number from placeholder
            num = re.search(r"\d+", placeholder)
            if num and int(num.group()) <= num_fewshot:
                valid_placeholders.add(placeholder)

        if not valid_placeholders:
            raise ValueError(
                f"Prompt template must contain valid placeholders for {num_fewshot} examples "
                f"using either the @ format or numbered format (e.g., <input1>, <output1>)."
            )

    def prepare(self) -> PromptResult:
        """Prepare a few-shot prompt using randomly selected seed data"""
        # Create a copy and shuffle to randomly select examples

        init_seed_data = self._init_seed_data.copy()
        generated_data = self.seed_data.copy()[: self._init_seed_data_len]

        selected_seed_examples = random.sample(init_seed_data, self.num_sample_from_seed_data)
        selected_gen_examples = random.sample(generated_data, self.num_fewshot - self.num_sample_from_seed_data)

        shuffled_data = selected_seed_examples + selected_gen_examples
        random.shuffle(shuffled_data)
        selected_examples = shuffled_data[: self.num_fewshot]

        # Create mapping for template replacement
        template_vars = {}
        for i, example in enumerate(selected_examples, 1):
            if "instruction" not in example or "response" not in example:
                raise ValueError(f"Example {i} missing required 'instruction' or 'response' key")
            # Support both @ style and numbered style placeholders
            input_placeholder_at = self.placeholder_formats["demonstration_input_placeholder"].replace("@", str(i))
            output_placeholder_at = self.placeholder_formats["demonstration_output_placeholder"].replace("@", str(i))
            input_placeholder_num = f"<input{i}>"
            output_placeholder_num = f"<output{i}>"

            template_vars[input_placeholder_at] = example["instruction"]
            template_vars[output_placeholder_at] = example["response"]
            template_vars[input_placeholder_num] = example["instruction"]
            template_vars[output_placeholder_num] = example["response"]

        # Replace all placeholders in the template
        prompt = self.prompt_template
        for placeholder, value in template_vars.items():
            prompt = prompt.replace(placeholder, value)

        return PromptResult(prompt=prompt, metadata={"num_examples": self.num_fewshot, "examples": selected_examples})


class ThemeBasedInstanceGenerationPromptLoader(InstanceGenerationPromptLoader):
    """Prompt loader that incorporates themes and triggers, inheriting from InstanceGenerationPromptLoader"""

    def __init__(
        self,
        prompt_template: str,
        seed_data: List[Dict[str, str]],
        num_fewshot: int = 3,
        num_sample_from_seed_data: int = 2,
        placeholder_formats: Optional[Dict[str, str]] = None,
        input_theme_list: Optional[List[str]] = None,
        first_word_list: Optional[List[str]] = None,
    ):
        """Initialize the theme-based instance generation prompt loader.

        Args:
            prompt_template: Template string for prompt generation
            seed_data: List of dictionaries containing instructions and responses
            num_fewshot: Number of examples to include in prompt
            placeholder_formats: Optional dictionary specifying custom placeholder formats
            num_sample_from_seed_data: Number of examples to sample from seed data
            input_theme_list: List of possible themes to choose from
            first_word_list: List of possible triggers to choose from
        """

        super().__init__(
            prompt_template=prompt_template,
            seed_data=seed_data,
            num_fewshot=num_fewshot,
            placeholder_formats=placeholder_formats,
            num_sample_from_seed_data=num_sample_from_seed_data,
        )
        self.input_theme_list = input_theme_list
        self.first_word_list = first_word_list

    def prepare(self) -> PromptResult:
        """Prepare a few-shot prompt using randomly selected seed data and themes/triggers"""
        # Get base prompt result from parent class
        prompt_result = super().prepare()
        prompt = prompt_result.prompt
        metadata = prompt_result.metadata

        # Add random theme if theme placeholder exists and theme list is not empty
        if self.placeholder_formats.get("input_theme") and self.input_theme_list:
            if self.placeholder_formats["input_theme"] in prompt:
                chosen_theme = random.choice(self.input_theme_list)
                prompt = prompt.replace(self.placeholder_formats["input_theme"], chosen_theme)
                metadata["chosen_theme"] = chosen_theme
        elif self.placeholder_formats.get("input_theme"):
            if self.placeholder_formats["input_theme"] not in prompt:
                raise ValueError(f"Prompt does not include input_theme placeholder")

        # Add random trigger if trigger placeholder exists and trigger list is not empty
        if self.placeholder_formats.get("first_word") and self.first_word_list:
            if self.placeholder_formats["first_word"] in prompt:
                chosen_trigger = random.choice(self.first_word_list)
                prompt = prompt.replace(self.placeholder_formats["first_word"], chosen_trigger)
                metadata["chosen_trigger"] = chosen_trigger
        elif self.placeholder_formats.get("first_word"):
            if self.placeholder_formats["first_word"] not in prompt:
                raise ValueError(f"Prompt does not include first_word placeholder")

        return PromptResult(prompt=prompt, metadata=metadata)


class ResponseGenerationPromptLoader(BasePromptLoader):
    """Prompt loader for response generation scenario"""

    def __init__(
        self,
        prompt_template: str,
        seed_data: List[Dict[str, str]],
        placeholder_formats: Optional[Dict[str, str]] = None,
    ):
        """Initialize the response generation prompt loader.

        Args:
            prompt_template: Template string for prompt generation
            seed_data: List of dictionaries containing instructions
            placeholder_formats: Optional dictionary specifying custom placeholder formats
        """
        self.prompt_template = prompt_template
        self.seed_data = seed_data
        self.placeholder_formats = placeholder_formats or {"test_input_placeholder": "<input>"}
        self._validate_placeholder(prompt_template)

    def _validate_placeholder(self, prompt_template: str):
        """Validate that the prompt template contains the required placeholder"""
        input_placeholder = self.placeholder_formats["test_input_placeholder"]
        if input_placeholder not in prompt_template:
            raise ValueError(
                f"Prompt template missing required placeholder: {input_placeholder}. "
                f"Template must contain the input placeholder."
            )

    def prepare(self, index: int = 0) -> PromptResult:
        """Prepare a prompt for the current instruction"""
        if index >= len(self.seed_data):
            raise StopIteration("No more seed data to process")

        current_item = self.seed_data[index]
        if "instruction" not in current_item:
            raise ValueError(f"Data item at index {index} missing 'instruction' key")

        prompt = self.prompt_template.replace(
            self.placeholder_formats["test_input_placeholder"], current_item["instruction"]
        )

        metadata = {"index": index, "instruction": current_item["instruction"]}

        return PromptResult(prompt=prompt, metadata=metadata)


class QualityEnhancementPromptLoader(BasePromptLoader):
    """Prompt loader for quality enhancement scenario"""

    def __init__(
        self,
        prompt_template: str,
        seed_data: List[Dict[str, str]],
        placeholder_formats: Optional[Dict[str, str]] = None,
    ):
        """Initialize the quality enhancement prompt loader.

        Args:
            prompt_template: Template string for prompt generation
            seed_data: List of dictionaries containing instructions and responses
            placeholder_formats: Optional dictionary specifying custom placeholder formats
        """
        self.prompt_template = prompt_template
        self.seed_data = seed_data
        self.placeholder_formats = placeholder_formats or {
            "test_input_placeholder": "<input>",
            "test_output_placeholder": "<output>",
        }
        self._validate_placeholder(prompt_template)

    def _validate_placeholder(self, prompt_template: str):
        """Validate that the prompt template contains the required placeholders"""
        input_placeholder = self.placeholder_formats["test_input_placeholder"]
        output_placeholder = self.placeholder_formats["test_output_placeholder"]
        if input_placeholder not in prompt_template or output_placeholder not in prompt_template:
            raise ValueError(
                f"Prompt template missing required placeholders: {input_placeholder} and/or {output_placeholder}. "
                f"Template must contain both input and output placeholders."
            )

    def prepare(self, index: int = 0) -> PromptResult:
        """Prepare a prompt for the current instruction-response pair"""
        if index >= len(self.seed_data):
            raise StopIteration("No more seed data to process")

        current_item = self.seed_data[index]
        if "instruction" not in current_item or "response" not in current_item:
            raise ValueError(
                f"Data item at index {index} missing required keys. " "Expected both 'instruction' and 'response'"
            )

        prompt = self.prompt_template
        prompt = prompt.replace(self.placeholder_formats["test_input_placeholder"], current_item["instruction"])
        prompt = prompt.replace(self.placeholder_formats["test_output_placeholder"], current_item["response"])

        metadata = {
            "index": index,
            "instruction": current_item["instruction"],
            "response": current_item["response"],
        }

        return PromptResult(prompt=prompt, metadata=metadata)


class JSONPromptLoader(InstanceGenerationPromptLoader):
    """Prompt loader that incorporates themes and triggers, inheriting from InstanceGenerationPromptLoader"""

    def __init__(
        self,
        prompt_template: str,
        seed_data: List[Dict[str, str]],
        num_fewshot: int = 3,
        num_sample_from_seed_data: int = 2,
        placeholder_formats: Optional[Dict[str, str]] = None,
    ):
        """Initialize the theme-based instance generation prompt loader.

        Args:
            prompt_template: Template string for prompt generation
            seed_data: List of dictionaries containing instructions and responses
            num_fewshot: Number of examples to include in prompt
            num_sample_from_seed_data: Number of examples to sample from seed data
            placeholder_formats: Optional dictionary specifying custom placeholder formats
        """

        super().__init__(
            prompt_template=prompt_template,
            seed_data=seed_data,
            num_fewshot=num_fewshot,
            placeholder_formats=placeholder_formats,
            num_sample_from_seed_data=num_sample_from_seed_data,
        )

    def prepare(self) -> PromptResult:
        """Prepare a few-shot prompt using randomly selected seed data and themes/triggers"""
        # Get base prompt result from parent class

    def prepare(self) -> PromptResult:
        """Prepare a few-shot prompt using randomly selected seed data"""
        # Create a copy and shuffle to randomly select examples

        init_seed_data = self._init_seed_data.copy()
        generated_data = self.seed_data.copy()[: self._init_seed_data_len]

        selected_seed_examples = random.sample(init_seed_data, self.num_sample_from_seed_data)
        selected_gen_examples = random.sample(generated_data, self.num_fewshot - self.num_sample_from_seed_data)

        shuffled_data = selected_seed_examples + selected_gen_examples
        random.shuffle(shuffled_data)
        selected_examples = shuffled_data[: self.num_fewshot]

        # Create mapping for template replacement
        template_vars = {}
        for i, example in enumerate(selected_examples, 1):
            if "instruction" not in example or "response" not in example:
                raise ValueError(f"Example {i} missing required 'instruction' or 'response' key")
            # Support both @ style and numbered style placeholders
            input_placeholder_at = self.placeholder_formats["demonstration_input_placeholder"].replace("@", str(i))
            output_placeholder_at = self.placeholder_formats["demonstration_output_placeholder"].replace("@", str(i))
            input_placeholder_num = f"<input{i}>"
            output_placeholder_num = f"<output{i}>"

            template_vars[input_placeholder_at] = json.dumps(example["instruction"])
            template_vars[output_placeholder_at] = json.dumps(example["response"])
            template_vars[input_placeholder_num] = json.dumps(example["instruction"])
            template_vars[output_placeholder_num] = json.dumps(example["response"])

        # Replace all placeholders in the template
        prompt = self.prompt_template
        for placeholder, value in template_vars.items():
            prompt = prompt.replace(placeholder, value)

        return PromptResult(prompt=prompt, metadata={"num_examples": self.num_fewshot, "examples": selected_examples})


def load_prompt_loader(prompt_loader_type: str, **kwargs) -> BasePromptLoader:
    if prompt_loader_type == "instance_generation":
        return InstanceGenerationPromptLoader(**kwargs)
    elif prompt_loader_type == "response_generation":
        return ResponseGenerationPromptLoader(**kwargs)
    elif prompt_loader_type == "quality_enhancement":
        return QualityEnhancementPromptLoader(**kwargs)
    else:
        raise ValueError(f"Invalid prompt loader type: {prompt_loader_type}")
