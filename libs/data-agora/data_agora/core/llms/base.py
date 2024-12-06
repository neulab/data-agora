# TODO: Implement abstract LLM class and mock llm class
from typing import Dict, List

from tqdm.auto import tqdm


class LLM:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.provider = "None"

    def chat(self, messages: List[Dict[str, str]], **kwargs):
        raise NotImplementedError("Chat method must be implemented by the subclass")


# TODO: Unify the return type of the chat method as OpenAI Chat Completion object

# OPENAI CHAT COMPLETION OBJECT
# {
#   "id": "chatcmpl-123456",
#   "object": "chat.completion",
#   "created": 1728933352,
#   "model": "gpt-4o-2024-08-06",
#   "choices": [
#     {
#       "index": 0,
#       "message": {
#         "role": "assistant",
#         "content": "Hi there! How can I assist you today?",
#         "refusal": null
#       },
#       "logprobs": null,
#       "finish_reason": "stop"
#     }
#   ],
# }


class MockLLM(LLM):
    def __init__(self, model_name: str = "mock", api_key: str = "mock", api_base: str = None):
        """Initialize the MockLLM with basic configurations."""
        super().__init__(model_name=model_name)
        self.api_base = api_base
        self.api_key = api_key

    def validate_mock(self):
        return True

    def chat(self, messages: List[Dict[str, str]], **kwargs):
        # messages: List of list of dictionaries containing messages
        assert isinstance(messages, list), "Messages must be provided as a list"

        for msg in messages:
            assert isinstance(msg, dict), "Each message must be provided as a dictionary"
            assert "role" in msg, "Each message must contain 'role' key"
            assert "content" in msg, "Each message must contain 'content' key"

        # Always return "Hello World" for each conversation
        return {
            "model": "mock",
            "choices": [{"message": {"role": "assistant", "content": "Hello World"}, "finish_reason": "stop"}],
        }


def _test():
    """Test function for MockLLM."""
    model = MockLLM()
    batch_messages = [
        [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "good morning?"},
        ]
    ] * 3

    for message in batch_messages:
        response = model.chat(message)
        print(response)


if __name__ == "__main__":
    _test()
