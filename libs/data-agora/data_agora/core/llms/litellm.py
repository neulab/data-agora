from typing import Dict, List

import litellm
from litellm import completion

from .base import LLM


class LiteLLM(LLM):
    def __init__(self, model_name: str, api_key: str, remove_stop: bool = False, api_base: str = None):
        """Initialize the LiteLLM with basic configurations."""
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.remove_stop = remove_stop
        self.provider = "litellm"
        # litellm.set_verbose=True

    def validate_litellm(self):
        return True

    def chat(self, messages: List[Dict[str, str]], **kwargs):
        assert isinstance(messages, list), "Messages must be provided as a list"
        assert all(isinstance(msg, dict) for msg in messages), "Each message must be provided as a dictionary"
        assert all("role" in msg for msg in messages), "Each message must contain 'role' key"
        assert all("content" in msg for msg in messages), "Each message must contain 'content' key"

        if self.remove_stop:
            kwargs.pop("stop", None)
        return completion(
            model=self.model_name, base_url=self.api_base, api_key=self.api_key, messages=messages, **kwargs
        )


def _test():
    """Test function for LiteLLM."""
    model = LiteLLM("openrouter/openai/gpt-3.5-turbo")
    messages = [{"role": "user", "content": "good morning?"}]

    responses = model.chat(messages)
    print(responses)

    return responses


if __name__ == "__main__":
    _test()
