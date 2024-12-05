from typing import Dict, List

from openai import OpenAI

from .base import LLM


class OpenAILLM(LLM):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        api_base: str = "https://api.openai.com/v1",
    ):
        self.model_name = model_name
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.provider = "openai"

    def chat(self, messages: List[Dict[str, str]], **kwargs):
        assert isinstance(messages, list), "Messages must be provided as a list"
        assert all(isinstance(msg, dict) for msg in messages), "Each message must be provided as a dictionary"
        assert all("role" in msg for msg in messages), "Each message must contain 'role' key"
        assert all("content" in msg for msg in messages), "Each message must contain 'content' key"

        try:
            return self.client.chat.completions.create(model=self.model_name, messages=messages, **kwargs)
        except Exception as e:
            print(f"Error in OpenAI API call: {e}")
            raise e
