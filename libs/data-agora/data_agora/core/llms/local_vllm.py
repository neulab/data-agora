from typing import Dict, List, Union


class VLLMError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)


try:
    from vllm import LLM, SamplingParams
except ImportError as e:
    raise VLLMError(
        status_code=1,
        message="Failed to import 'vllm' package. Make sure it is installed correctly.",
    ) from e


class LocalVLLM:
    def __init__(
        self,
        model: str,
        **vllm_kwargs,
    ) -> None:
        self.model_name: str = model

        self.model = LLM(
            model=model,
            **vllm_kwargs,
        )

    def validate_vllm(self):
        return True

    def chat(self, messages: List[Dict[str, str]], **kwargs):
        for message in messages:
            if not isinstance(message, list):
                assert 0, "Each message must be provided as a list"
            for msg in message:
                if not isinstance(msg, dict):
                    assert 0, "Each message must be provided as a dictionary"
                if "role" not in msg:
                    assert 0, "Each message must contain 'role' key"
                if "content" not in msg:
                    assert 0, "Each message must contain 'content' key"

        sampling_params = SamplingParams(**kwargs.pop("sampling_params", {}))
        return self.model.chat(messages=messages, sampling_params=sampling_params, **kwargs)
