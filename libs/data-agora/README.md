<p align="center">
  <img src="https://raw.githubusercontent.com/neulab/data-agora/main/assets/agorabench.png" alt="Agora-Logo" style="width: 50%; display: block; margin: auto;">
</p>

<h1 align="center">🏛️ Agora 🏛️</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2412.03679"><img src="https://img.shields.io/badge/arXiv-2412.03679-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/Data-Agora"><img src="https://img.shields.io/badge/Hugging%20Face-Organization-ff9d00" alt="Hugging Face Organization"></a>
  <a href="https://github.com/neulab/data-agora/blob/main/LICENSE"><img src="https://img.shields.io/github/license/neulab/data-agora.svg" alt="License"></a>
  <a href="https://pypi.org/project/data-agora/"><img src="https://badge.fury.io/py/data-agora.svg" alt="PyPI version"></a>
</p>

<p align="center">
  ⚡ A repository for generating synthetic data with LLMs & evaluating LLMs' data generation capabilities 🚀 ⚡ <br>
</p>


## **Latest News** 🔥

- [2024/12] We release the **Agora** and **Agora-Bench**!
  - **Agora-Bench** covers 9 settings, measuring data generation capabilities across 3 domains and 3 data generation methods.
  - **Agora** is an easily customizable framework for data generation with LLMs.
  - Checkout our [dataset](https://huggingface.co/Data-Agora), [checkpoints](https://huggingface.co/Data-Agora), [leaderboard](https://huggingface.co/spaces/prometheus-eval/BiGGen-Bench-Leaderboard), and the [code](https://github.com/neulab/data-agora)!

## What does Agora mean?

<p align="center">
  <img src="https://raw.githubusercontent.com/neulab/data-agora/main/assets/agora.png" alt="Agora-Logo" style="width: 80%; display: block; margin: auto;">
</p>

*In ancient Athens, the Agora was a public space where citizens would gather to debate, share news, learn from each other, and listen to famous philosophers.*

We made an analogy between data generators and teachers, where different generators teach student models using synthetic data in AgoraBench!


## 🔧 Installation

Installation with pip:

```shell
pip install data-agora
```

## Project Structure 📁

### Root Directory
```
.
├── agora_scripts/           # Scripts for converting and handling data formats
│   ├── prompts/            # Various prompt templates
│   └── run.py             # Main execution script
├── assets/                 # Project images and visual assets
├── libs/                   # Core libraries
│   └── data-agora/        # Main data processing library
│       ├── data_agora/    # Core data agora implementation
│       │   ├── core/      # Core functionality (LLMs, parsers, validators)
├── train/                  # Training related code (based on llama-recipes)
└── LICENSE
```

#### data-agora Library (`libs/data-agora/`)
- Core implementation for data processing and handling
- Includes LLM integrations (OpenAI, vLLM, etc.)
- Parsers and validators for data processing
- Serving capabilities for deployment

#### Agora Scripts (`agora_scripts/`)
- Tools for data format conversion
- Collection of prompt templates for different use cases
- Main execution script for running the pipeline

#### Training (`train/`)
- Based on Meta's [llama-recipes](https://github.com/meta-llama/llama-recipes/tree/main) repository
- Contains training configurations and utilities

## Usage Guide 🚀

Our library is convenient for two types of audiences:
1. **Testing an LM's Data Generation Capability with AgoraBench**: Using the pre-built pipeline, you can easily measure the data generation capabilities of different LLMs.
2. **Custom Usage**: You could customize the pipeline for your own tasks to generate large amounts of synthetic data.

## **Testing an LM's Data Generation Capability with AgoraBench**

### Step 1: Generate Data with Pre-built Pipeline
You could simply run the following script:
```
cd "./alchemy_scripts"

python3 run.py --method "instance_generation" --domain "math" --model_name "gpt-4o-mini-2024-07-18" --max_tokens 4096 --temperature 1.0 --num_instances 10000 --num_threads 4 --api_key ""
```
- method should be either "instance_generation", "response_generation", or "quality_enhancement".
- domain should be either "math", "general", "code'.
- model_name should be exactly the same with how you call it on OpenAI API, LiteLLM, or vLLM.

- The resulting dataset should look as follows:
```
[
   {
      "config": "",
      "instruction": "",
      "response": ""
   },
   [...]
]
```

### Step 2: Upload the dataset to huggingface
You could use the following function:
```
from datasets import DatasetDict

def upload_to_huggingface(data, dataset_name, hf_key):
    dataset = Dataset.from_list(data)
    dataset_dict = DatasetDict({"train": dataset})
    api = HfApi()
    dataset_dict.push_to_hub(dataset_name, token=hf_key, private=True)
```
  

### Step 3: Train Student Models with Synthetic Data
The following code is modified based on Meta's [llama-recipes](https://github.com/meta-llama/llama-recipes)!

First, install the required packages
```
cd ./llama-recipes
pip3 install -r requirements.txt
pip3 install -e .
pip3 install wandb
wandb login
huggingface-cli login
```

Then, launch the following code.
```
gpu = 4
lr = 1e-5
checkpoint_dir = ""
hf_cache_dir = ""
hf_dataset_name = ""

torchrun --nnodes 1 --nproc_per_node $gpu \
        src/llama_recipes/finetuning.py \
        --model_name meta-llama/Meta-Llama-3.1-8B \
        --dist_checkpoint_root_folder "${checkpoint_dir}" \
        --dist_checkpoint_folder "${hf_dataset_name}" \
        --hf_cache_dir "${hf_cache_dir}" \
        --dataset "$hf_dataset_name" \
        --run_validation True \
        --context_length 4096 \
        --gradient_accumulation_steps 8 \
        --batching_strategy "packing" \
        --use_fast_kernels \
        --enable_fsdp \
        --pure_bf16 \
        --low_cpu_fsdp \
        --batch_size_training 2 \
        --num_epochs $num_epochs \
        --lr $lr \
        --weight_decay 0.01 \
        --use_wandb
```
- You have to fill in:
  - checkpoint_dir (where the checkpoint is saved)
  - hf_cache_dir (where huggingface cache is saved)
  - hf_dataset_name (the dataset you uploaded on hf from Stage 1)

- For uploading the checkpoint to huggingface, you could refer to this [code](https://github.com/neulab/data-agora/blob/main/llama-recipes/src/llama_recipes/convert_fsdp_to_hf.py).


### Step 5: Evaluate Student Models and Measure Performance Gap Recovered (PGR)
For evaluating the trained student models, we used the following libraries:
- **AlpacaEval 2.0 (Instruction-following)**: [link](https://github.com/tatsu-lab/alpaca_eval)
- **Arena-Hard (Instruction-following)**: [link](https://github.com/lmarena/arena-hard-auto)
- **MBPP (Code)**: [link](https://github.com/evalplus/evalplus)
- **Human-Eval (Code)**: [link](https://github.com/evalplus/evalplus)

For **GSM8K (Math)** and **MATH (Math)**, we implemented our custom code:
TO BE ADDED



## **Custom Usage**
For custom usage with different pipelines, parsing mechanisms, and validation logics, Alchemy supports convenient customization through abstract classes.

### **Prompt Loader**: A class that prepares the meta-prompt passed to the data generator.
```python
class CustomPromptLoader(InstanceGenerationPromptLoader):
   def __init__(self, prompt_template: str, seed_data: List[Dict], num_fewshot: int, placeholder_formats: Dict[str, str] = None, num_sample_from_seed_data: Optional[int] = None, [...]):
      super().__init__(prompt_template, seed_data, num_fewshot, placeholder_formats, num_sample_from_seed_data)
      [...]
    
    def prepare(self) -> PromptResult:
      [...]
      return PromptResult(prompt=prompt, metadata=metadata)
```

### **Parser**: A class that separates the instruction and response from the data generator's output.
```python
class CustomParser(Parser):

   def parse(self, prompt, teacher_model_output, placeholder_formats, [...]):
      [...]
      return {"instruction: instruction, "response": response}
```

### **Validator**: A class that determines if the output is valid or not.
```python
class CustomValidator(Validator):
   def validate(self, instruction: str, response: str, [...]):
      [...]
      if [...]:
        return True
      else:
        return False
```

### **All together**

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

agora = Agora(
    llm=llm,
    placeholder_formats=placeholder_formats,
    prompt_loader=prompt_loader,
    parser=parser,
    validator=validator,
    sampling_params=sampling_params
)

# Use cache_file to resume from previous results: The Alchemy class will automatically make a cache file "final_result.jsonl" for example
result = agora.run(num_instances=10000, num_threads=16, output_file="./results/final_result.json")
print(result[0])
```

## Citation

If you find our work useful, please consider citing our paper!

```bibtex
@misc{kim2024evaluating,
      title={Evaluating Language Models as Synthetic Data Generators}, 
      author={Seungone Kim and Juyoung Suk and Xiang Yue and Vijay Viswanathan and Seongyun Lee and Yizhong Wang and Kiril Gashteovski and Carolin Lawrence and Sean Welleck and Graham Neubig},
      year={2024},
      eprint={2412.03679},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.03679}, 
}
```
