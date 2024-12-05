<p align="center">
  <img src="assets/agora.png" alt="Agora-Logo" style="width: 70%; display: block; margin: auto;">
</p>

<h1 align="center">🏛️ Agora 🏛️</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2405.01535"><img src="https://img.shields.io/badge/arXiv-2405.01535-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/prometheus-eval"><img src="https://img.shields.io/badge/Hugging%20Face-Organization-ff9d00" alt="Hugging Face Organization"></a>
  <a href="https://github.com/prometheus-eval/prometheus-eval/blob/main/LICENSE"><img src="https://img.shields.io/github/license/prometheus-eval/prometheus-eval.svg" alt="License"></a>
  <a href="https://pypi.org/project/prometheus-eval/"><img src="https://badge.fury.io/py/prometheus-eval.svg" alt="PyPI version"></a>
</p>

<p align="center">
  ⚡ A repository for generating data with LLMs & evaluating LLMs' data generation capabilities 🚀 ⚡ <br>
</p>


**Latest News** 🔥

- [2024/12] We release the **Agora** and **Agora-Bench**!
  - **Agora-Bench** covers 9 settings, measuring data generation capabilities across 3 domains and 3 data generation methods.
  - **Agora** is an easily customizable framework for data generation with LLMs.
  - Checkout our [dataset](https://huggingface.co/datasets/prometheus-eval/BiGGen-Bench), [checkpoints](https://huggingface.co/datasets/prometheus-eval/BiGGen-Bench-Results), [leaderboard](https://huggingface.co/spaces/prometheus-eval/BiGGen-Bench-Leaderboard), and the [code](https://github.com/prometheus-eval/prometheus-eval/tree/main/BiGGen-Bench)!


## 🔧 Installation

Installation with pip:

```shell
pip install data-agora
```

## Project Structure 📁

```
agora/
├── core/                   # Core framework components
│   ├── llms/               # LLM implementations
│   │   ├── base.py         # Abstract LLM interface
│   │   ├── litellm.py      # LiteLLM integrationå
│   │   ├── openai.py       # OpenAI API integration
│   │   ├── test.py         # Test LLM implementation
│   │   └── vllm.py         # vLLM integration (to be implemented)
│   ├── parsers.py          # Parsing teacher model's output into instruction-response pairs
│   ├── prompt_loaders.py   # Prompt preparation
│   └── validators.py       # Validating the instruction-response pairs
└── agora.py                # Main class orchestrating the pipeline
```

## Usage Guide 🚀

### **Using Pre-built Pipeline**

To use AlchemyBench for replicating the results from the paper or using the exact same pipeline for custom use with potentially different seed data:
```
cd "./alchemy_scripts"

python3 run.py --method {} --domain {} --model_name {} --max_tokens 4096 --temperature 1.0 --num_instances 50 --num_threads 4
```
- method should be either "instance_generation", "response_generation", or "quality_enhancement" For other custom pipelines, refer to the Section below.
- domain should be either "math", "general", "code'. When using custom data and there is no distinct constraint of how the data should look like, use "general".
- model_name should be exactly the same with how you call it on OpenAI API, LiteLLM, or vLLM.


### **Custom Usage**

For custom usage with different pipelines, parsing mechanisms, and validation logics, Alchemy supports convenient customization through abstract classes.

1.**Prompt Loader:**
```python
class CustomPromptLoader(InstanceGenerationPromptLoader):
   def __init__(self, prompt_template: str, seed_data: List[Dict], num_fewshot: int, placeholder_formats: Dict[str, str] = None, num_sample_from_seed_data: Optional[int] = None, [...]):
      super().__init__(prompt_template, seed_data, num_fewshot, placeholder_formats, num_sample_from_seed_data)
      [...]
    
    def prepare(self) -> PromptResult:
      [...]
      return PromptResult(prompt=prompt, metadata=metadata)
```

2.**Parser:**
```python
class CustomParser(Parser):

   def parse(self, prompt, teacher_model_output, placeholder_formats, [...]):
      [...]
      return {"instruction: instruction, "response": response}
```

3.**Validator:**
```python
class CustomValidator(Validator):
   def validate(self, instruction: str, response: str, [...]):
      [...]
      if [...]:
        return True
      else:
        return False
```

4.**Data Generation with Agora:**

Then, you could write a script that utilizes the custom classes to generate data.

```python
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

alchemy = Alchemy(
    llm=llm,
    placeholder_formats=placeholder_formats,
    prompt_loader=prompt_loader,
    parser=parser,
    validator=validator,
    sampling_params=sampling_params
)

# Use cache_file to resume from previous results: The Alchemy class will automatically make a cache file "final_result.jsonl" for example
result = alchemy.run(num_instances=10000, num_threads=16, output_file="./results/final_result.json")
print(result[0])
```
