import agora
from agora.core.llms.openai import OpenAILLM
from agora.core.prompt_loaders import load_prompt_loader, PromptResult, ThemeBasedInstanceGenerationPromptLoader, InstanceGenerationPromptLoader, ResponseGenerationPromptLoader, QualityEnhancementPromptLoader
from agora.core.parsers import InstanceGenerationParser, ResponseGenerationParser, QualityEnhancementParser
from agora.core.validators import MathValidator, CodeValidator, GeneralValidator
from agora.agora import Agora, AgoraConfig
import json
import argparse


# MODIFY THE PLACEHOLDER FORMATS BASED ON YOUR PROMPT TEMPLATE
# Demonstration related placeholders are only used for instance generation
# Input Theme place holder is an example of a custom placeholder
placeholder_formats = {
    "demonstration_input_placeholder": "<input@>",
    "demonstration_output_placeholder": "<output@>",
    "test_input_placeholder": "<input>",
    "test_output_placeholder": "<output>",
    "test_input_trigger": "INPUT:",
    "test_output_trigger": "OUTPUT:",
    "stop_phrase": "[END]",
    "input_theme": "<input_theme>",
}


with open("", "r") as f:
    seed_data = json.load(f)

with open("", "r") as f:
    prompt_template = f.read()

llm = OpenAILLM(model_name="gpt-4o-mini-2024-07-18", api_key="")

prompt_loader = CustomPromptLoader(prompt_template=prompt_template, seed_data=seed_data, num_fewshot=3, placeholder_formats=placeholder_formats, num_sample_from_seed_data=2)
parser = CustomParser()
validator = CustomValidator()


sampling_params = {
    "max_tokens": args.max_tokens,
    "temperature": args.temperature,
    "top_p": 0.9,
    "stop": placeholder_formats["stop_phrase"]
}

agora = Agora(
    llm=llm,
    placeholder_formats=placeholder_formats,
    prompt_loader=prompt_loader,
    parser=parser,
    validator=validator,
    sampling_params=sampling_params
)

# Use cache_file to resume from previous results: The Agora class will automatically make a cache file "final_result.jsonl" for example
result = agora.run(num_instances=10000, num_threads=16, output_file="./results/final_result.json")
print(result[0])
    
