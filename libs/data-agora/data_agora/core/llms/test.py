import random
import string
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from data_agora.core.llms.base import LLM


class TestLLM(LLM):
    """Test LLM that generates domain-specific mock responses"""

    def __init__(self, model_name: str = "test"):
        super().__init__(model_name=model_name)
        self.valid_prob = 1.0  # Probability of generating valid responses

    def chat(self, messages: List[Dict[str, str]], **kwargs):
        """Generate test responses based on domain"""
        # Extract generation parameters
        max_tokens = kwargs.get("max_tokens", 4096)
        temperature = kwargs.get("temperature", 1.0)
        domain = kwargs.get("domain", "general")

        # Generate random instruction
        random_instruction = "".join(random.choices(string.ascii_letters + " ", k=random.randint(20, 50))) + "."

        # Determine if response should be valid
        is_valid = random.random() < self.valid_prob

        completion, finish_reason = self._generate_response(
            domain=domain, random_instruction=random_instruction, is_valid=is_valid
        )

        # Simulate token usage
        usage = {
            "prompt_tokens": len(str(messages)) // 4,
            "completion_tokens": len(completion) // 4,
            "total_tokens": (len(str(messages)) + len(completion)) // 4,
        }

        return {
            "model": "test",
            "choices": [{"message": {"role": "assistant", "content": completion}, "finish_reason": finish_reason}],
        }

    def _generate_response(self, domain: str, random_instruction: str, is_valid: bool) -> Tuple[str, str]:
        """Generate domain-specific test responses"""
        if is_valid:
            return self._generate_valid_response(domain, random_instruction)
        else:
            return self._generate_invalid_response(domain, random_instruction)

    def _generate_valid_response(self, domain: str, random_instruction: str) -> Tuple[str, str]:
        """Generate valid responses for different domains"""
        if domain == "general":
            completion = (
                f"INPUT: {random_instruction}\n"
                f"OUTPUT: {''.join(random.choices(string.ascii_letters + ' ', k=random.randint(20, 50)))}.\n"
                "[END]"
            )
        elif domain == "code":
            completion = (
                f"INPUT: {random_instruction}\n"
                f"OUTPUT: ```python\n"
                f"def test_function():\n"
                f"    return '{''.join(random.choices(string.ascii_letters, k=5))}'\n"
                "```\n"
                "[END]"
            )
        elif domain == "math":
            completion = (
                f"INPUT: {random_instruction}\n"
                f"OUTPUT: Let's solve this step by step:\n"
                f"1. {''.join(random.choices(string.ascii_letters + ' ', k=20))}\n"
                f"[RESULT]{''.join(random.choices(string.digits + '.', k=5))}[/RESULT]\n"
                "[END]"
            )
        return completion, "stop"

    def _generate_invalid_response(self, domain: str, random_instruction: str) -> Tuple[str, str]:
        """Generate invalid responses for different domains"""
        invalid_templates = {
            "general": [
                # Missing period at end
                lambda: (
                    f"INPUT: {random_instruction}\n"
                    f"OUTPUT: {''.join(random.choices(string.ascii_letters + ' ', k=20))}\n"
                    "[END]"
                ),
                # Missing [END] token
                lambda: (
                    f"INPUT: {random_instruction}\n"
                    f"OUTPUT: {''.join(random.choices(string.ascii_letters + ' ', k=20))}."
                ),
                # Multiple OUTPUT markers
                lambda: (
                    f"INPUT: {random_instruction}\n" "OUTPUT: First output.\n" "OUTPUT: Second output.\n" "[END]"
                ),
                # No OUTPUT marker
                lambda: (
                    f"INPUT: {random_instruction}\n"
                    f"{''.join(random.choices(string.ascii_letters + ' ', k=20))}.\n"
                    "[END]"
                ),
                # Very long response
                lambda: (
                    f"INPUT: {random_instruction}\n"
                    f"OUTPUT: {''.join(random.choices(string.ascii_letters + ' ', k=20000))}.\n"
                    "[END]"
                ),
            ],
            "code": [
                # Missing closing backticks
                lambda: (
                    f"INPUT: {random_instruction}\n"
                    "OUTPUT: ```python\n"
                    "def test_function():\n"
                    "    return 'test'\n"
                    "[END]"
                ),
                # Invalid language
                lambda: (
                    f"INPUT: {random_instruction}\n"
                    "OUTPUT: ```invalid_lang\n"
                    "def test_function():\n"
                    "    return 'test'\n"
                    "```\n"
                    "[END]"
                ),
                # Contains forbidden keywords
                lambda: (
                    f"INPUT: {random_instruction}\n"
                    "OUTPUT: ```python\n"
                    "# example usage\n"
                    "def test_function():\n"
                    "    return 'test'\n"
                    "```\n"
                    "[END]"
                ),
            ],
            "math": [
                # Missing [RESULT] tags
                lambda: (
                    f"INPUT: {random_instruction}\n"
                    "OUTPUT: Let's solve this:\n"
                    f"{''.join(random.choices(string.digits + '.', k=5))}\n"
                    "[END]"
                ),
                # Multiple [RESULT] tags
                lambda: (
                    f"INPUT: {random_instruction}\n"
                    "OUTPUT: Step 1\n"
                    "[RESULT]5[/RESULT]\n"
                    "Step 2\n"
                    "[RESULT]10[/RESULT]\n"
                    "[END]"
                ),
                # No [RESULT] tags
                lambda: (f"INPUT: {random_instruction}\n" "OUTPUT: The answer is 42\n" "[END]"),
            ],
        }

        template_fn = random.choice(invalid_templates[domain])

        return template_fn(), "length"


def _test():
    """Test function for TestLLM."""
    model = TestLLM()
    messages = [
        {"role": "system", "content": "You are a data generator."},
        {"role": "user", "content": "Generate a test response."},
    ]

    # Test for each domain
    for domain in ["general", "code", "math"]:
        print(f"\nTesting {domain} domain:")
        content, response = model.chat(messages, domain=domain)
        print(f"Completion:\n{content}")
        print(f"Finish reason: {response['finish_reason']}")
        print(f"Usage: {response['usage']}")


if __name__ == "__main__":
    _test()
