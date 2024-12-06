import json
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from transformers import AutoTokenizer


class Parser(ABC):
    """Abstract base class for validating model outputs"""

    @abstractmethod
    def parse(
        self, prompt, teacher_model_output, placeholder_formats: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Parse the instruction and response from model output

        Args:
            teacher_model_output: OpenAI chat teacher_model_output object

        Returns:
            Dictionary containing instruction and response
        """
        pass


class InstanceGenerationParser(Parser):
    """Parser for instance generation scenario"""

    def parse(self, prompt, teacher_model_output, placeholder_formats: Dict[str, str]) -> Dict[str, str]:

        instruction = (
            teacher_model_output.split(placeholder_formats["test_input_trigger"])[-1]
            .split(placeholder_formats["test_output_trigger"])[0]
            .strip()
        )
        response = (
            teacher_model_output.split(placeholder_formats["test_output_trigger"])[-1]
            .split(placeholder_formats["stop_phrase"])[0]
            .strip()
        )

        return {"instruction": instruction, "response": response}


class ResponseGenerationParser(Parser):
    """Parser for response generation scenario"""

    def parse(self, prompt, teacher_model_output, placeholder_formats: Dict[str, str]) -> Dict[str, str]:
        instruction = (
            prompt.split(placeholder_formats["test_input_trigger"])[-1]
            .split(placeholder_formats["test_output_trigger"])[0]
            .strip()
        )
        response = (
            teacher_model_output.split(placeholder_formats["test_output_trigger"])[-1]
            .split(placeholder_formats["stop_phrase"])[0]
            .strip()
        )

        return {"instruction": instruction, "response": response}


class QualityEnhancementParser(Parser):
    """Parser for quality enhancement scenario"""

    def parse(self, prompt, teacher_model_output, placeholder_formats: Dict[str, str]) -> Dict[str, str]:
        instruction = (
            teacher_model_output.split(placeholder_formats["test_input_trigger"])[-1]
            .split(placeholder_formats["test_output_trigger"])[0]
            .strip()
        )
        response = (
            teacher_model_output.split(placeholder_formats["test_output_trigger"])[-1]
            .split(placeholder_formats["stop_phrase"])[0]
            .strip()
        )

        return {"instruction": instruction, "response": response}


class InstanceGenerationDictParser(Parser):
    """Parser for instance generation scenario"""

    def parse(self, prompt, teacher_model_output) -> Dict[str, str]:
        x = dict(teacher_model_output)

        instruction = x["instruction"]
        response = x["response"]

        return {"instruction": instruction, "response": response}


class ResponseGenerationDictParser(Parser):
    """Parser for response generation scenario"""

    def parse(self, prompt, teacher_model_output) -> Dict[str, str]:
        x = dict(teacher_model_output)

        instruction = x["instruction"]
        response = x["response"]

        return {"instruction": instruction, "response": response}


class QualityEnhancementDictParser(Parser):
    """Parser for quality enhancement scenario"""

    def parse(self, prompt, teacher_model_output) -> Dict[str, str]:
        x = dict(teacher_model_output)

        instruction = x["instruction"]
        response = x["response"]

        return {"instruction": instruction, "response": response}


class JSONParser(Parser):
    """Parser when LLM generates data as JSON strings"""

    def parse(self, prompt, teacher_model_output, placeholder_formats=None) -> Dict[str, str]:
        x = json.loads(teacher_model_output)

        instruction = x["instruction"]
        response = x["response"]

        return {"instruction": instruction, "response": response}
