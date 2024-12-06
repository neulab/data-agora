import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from transformers import AutoTokenizer


class Validator(ABC):
    """Abstract base class for validating model outputs"""

    @abstractmethod
    def validate(self, instruction: str, response: str) -> bool:
        """
        Validate the format of generated output and return parsed result if valid.

        Args:
            completion: OpenAI chat completion object

        Returns:
            Boolean indicating validity
        """
        pass


class GeneralValidator(Validator):
    """Validator for instance generation scenario"""

    def __init__(self, tokenizer_name: str, max_tokens: int, placeholder_formats: Optional[Dict[str, str]] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_tokens = max_tokens
        self.placeholder_formats = placeholder_formats

    def validate(self, instruction: str, response: str) -> bool:

        flg1 = validate_length(instruction, response, self.tokenizer, self.max_tokens)
        flg2 = True
        if self.placeholder_formats is not None:
            flg2 = validate_forbidden_keywords(response, list(self.placeholder_formats.values()))

        return flg1 and flg2


class MathValidator(Validator):
    """Validator for instance generation scenario"""

    def __init__(self, tokenizer_name: str, max_tokens: int, placeholder_formats: Optional[Dict[str, str]] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_tokens = max_tokens
        self.placeholder_formats = placeholder_formats

    def validate(self, instruction: str, response: str) -> bool:
        flg1 = validate_length(instruction, response, self.tokenizer, self.max_tokens)
        flg2 = validate_keywords(
            response, keyword_occurance={"[RESULT]": 1, "[/RESULT]": 1}, end_keywords=["[/RESULT]"]
        )
        flg3 = True
        if self.placeholder_formats is not None:
            flg3 = validate_forbidden_keywords(response, list(self.placeholder_formats.values()))

        return flg1 and flg2 and flg3


class CodeValidator(Validator):
    """Validator for instance generation scenario"""

    def __init__(self, tokenizer_name: str, max_tokens: int, placeholder_formats: Optional[Dict[str, str]] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_tokens = max_tokens
        self.placeholder_formats = placeholder_formats
        self.forbidden_keywords = [
            "example usage",
            "this function",
            "test the function",
            "test cases",
            "this implementation",
            "explanation",
            "this solution",
        ]

    def validate(self, instruction: str, response: str) -> bool:
        flg1 = validate_length(instruction, response, self.tokenizer, self.max_tokens)
        if response.count("```") % 2 == 1:
            flg2 = False
        else:
            flg2 = True
        # Extract text outside code blocks
        parts = response.split("```")
        text_outside_code = ""
        for i in range(0, len(parts), 2):
            text_outside_code += parts[i]
        flg3 = validate_forbidden_keywords(text_outside_code, self.forbidden_keywords)
        flg4 = validate_keywords(response, start_keywords=["```"], end_keywords=["```"])
        flg5 = True
        if self.placeholder_formats is not None:
            flg5 = validate_forbidden_keywords(response, list(self.placeholder_formats.values()))

        return flg1 and flg2 and flg3 and flg4 and flg5


def validate_keywords(
    text: str,
    keyword_occurance: Dict[str, int] = None,
    start_keywords: List[str] = None,
    end_keywords: List[str] = None,
) -> bool:
    """Validates presence/position of keywords in output"""

    # Check required keywords
    if keyword_occurance is not None:
        for keyword, count in keyword_occurance.items():
            if text.count(keyword) != count:
                return False

    # Check start keywords
    if start_keywords is not None:
        if not any(text.startswith(k) for k in start_keywords):
            return False

    # Check end keywords
    if end_keywords is not None:
        if not any(text.endswith(k) for k in end_keywords):
            return False

    return True


def validate_length(model_input: str, model_output: str, tokenizer, max_tokens: int) -> bool:
    """Validates output length"""

    # Need to change hard-coding for llama3.1
    def create_prompt_with_llama3_format(prompt):
        system_message = "You are a helpful AI assistant."
        formatted_text = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|>"
        )
        formatted_text += "<|start_header_id|>user<|end_header_id|>\n\n" + prompt + "<|eot_id|>"
        formatted_text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return formatted_text

    token_count = len(tokenizer.encode(create_prompt_with_llama3_format(model_input) + model_output + "<|eot_id|>"))
    return token_count <= max_tokens


def validate_forbidden_keywords(text: str, forbidden_keywords: List[str]) -> bool:
    """Validates absence of forbidden keywords"""
    text = text.lower()

    for keyword in forbidden_keywords:
        if keyword.lower() in text:
            return False

    return True


def validate_dict_format(model_output: str, required_keys: List[str]) -> bool:
    """Validates dictionary format and required keys"""
    try:
        # Parse instruction and response
        parts = model_output.split("\nOUTPUT: ")
        instruction = parts[0].split("INPUT: ")[-1].strip()
        response = parts[1].strip()

        result = {"instruction": instruction, "response": response}

        # Verify all required keys exist
        for key in required_keys:
            if key not in result:
                return False

        return True

    except:
        return False
