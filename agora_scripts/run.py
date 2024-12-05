import agora
# from agora.core.llms.test import TestLLM
from agora.core.llms.litellm import LiteLLM
from agora.core.llms.openai import OpenAILLM
from agora.core.prompt_loaders import load_prompt_loader, PromptResult, ThemeBasedInstanceGenerationPromptLoader, InstanceGenerationPromptLoader, JSONPromptLoader, ResponseGenerationPromptLoader, QualityEnhancementPromptLoader
from agora.core.parsers import InstanceGenerationParser, ResponseGenerationParser, QualityEnhancementParser, JSONParser
from agora.core.validators import MathValidator, CodeValidator, GeneralValidator
from agora.agora import Agora, AgoraConfig
import json
import argparse

general_themes = [
    "Health", 
    "Professional", 
    "Linguistics", 
    "Entertainment", 
    "Technology",
    "Literature", 
    "Implementing a Code",
    "Software Development",
    "History",
    "Exercise",
    "Language Learning",
    "Science", 
    "Gaming", 
    "Philosophy", 
    "Social Studies", 
    "Travel", 
    "Art", 
    "Sports", 
    "Mathematics", 
    "Social Interaction", 
    "DIY Projects", 
    "Cooking", 
    "Technical Writing", 
    "Recommendations", 
    "Creative Writing", 
    "How-To Style Question and Answers", 
    "Factual Question Answering", 
    "Puzzles and Logical Reasoning"
]

general_triggers = [
    "Write ",
    "Write a function ",
    "def ",
    "Make me ",
    "Can you ",
    "Will you ",
    "Would you ",
    "Use ",
    "Solve this ",
    "Explain ",
    "Design ",
    "If I have ",
    "I have ",
    "I need ",
    "I want ",
    "I'm ",
    "I've ",
    "I'd ",
    "If you ",
    "We ",
    "My ",
    "Describe how ",
    "Describe ",
    "How ",
    "Why ",
    "What ",
    "When ",
    "Where ",
    "Which ",
    "Who ",
    "Classify the following ",
    "Determine the following ",
    "Implement ",
    "Consider the ",
    "Prove that ",
    "Considering that ",
    "Introduce ",
    "### Problem:  ",
    "Fix ",
    "Improve ",
    "Optimize ",
    "Clean this ",
    "Correct ",
    "In what ",
    "Given a ",
    "Given that ",
    "Given the following ",
    "Help me ",
    "Hello ",
    "You are a ",
    "Is there ",
    "Give me ",
    "Here is ",
    "Here are ",
    "A ",
    "An ",
    "My ",
    "Your ",
    "You're ",
    "Rewrite ",
    "Show me ",
    "Show that ",
    "Act as ",
    "Devise ",
    "Your goal ",
    "Suppose ",
    "Imagine ",
    "There ",
    "This ",
    "That ",
    "These ",
    "The ",
    "In after ",
    "1.  ",
    "List ",
    "Analyze ",
    "Compare ",
    "Evaluate ",
    "Assess ",
    "Identify ",
    "Search ",
    "Find ",
    "In ",
    "Pretend ",
    "Conduct ",
    "During ",
    "In the ",
    "In the past ",
    "In the future ",
    "Using ",
    "As part of ",
    "As a result of ",
    "As a consequence of ",
    "In order to ",
    "Tell me ",
    "Now ",
    "Count ",
    "Summarize ",
    "Query ",
    "Extract ",
    "### ",
    ">> ",
    "``` ",
    "Prepare ",
    "import ",
    "Visualize ",
    "Generate ",
    "Provide ",
    "Read the ",
    "Take the ",
    "Go ",
    "Act as ",
    "Is ",
    "Are ",
    "Were ",
    "Suggest ",
    "Recommend ",
    "Contrast ",
    "Look for ",
    "Define ",
    "As a "
    "Question: ",
    "Express ",
    "Let "
]

code_themes = [
    "Implementing a Python Code",
    "Implementing a Java Code",
    "Implementing a C++ Code",
    "Implementing a C Code",
    "Implementing a JavaScript Code",
    "Implementing a C# Code",
    "Implementing a PHP Code",
    "Fixing a Python Code",
    "Fixing a Java Code",
    "Fixing a C++ Code",
    "Fixing a C Code",
    "Fixing a JavaScript Code",
    "Fixing a C# Code",
    "Fixing a PHP Code"
]

math_themes = [
    "Algebra",
    "Counting and Probability",
    "Geometry",
    "Number Theory",
    "Calculus",
    "Conic sections",
    "Group Theory and Ring Theory",
    "Field Theory and Vector Spaces",
    "Differential Equations",
    "Linear Algebra",
    "Real Analysis",
    "Complex Analysis",
    "Combinatorics",
    "Topology",
    "Trigonometry",
    "Combinatorics",
    "Graph Theory",
    "Set Theory",
    "Math Word Problems with Polynomial GCD",
    "Math Word Problems with Probability and Number Theory",
    "Math Word Problems with Matrices",
    "Math Word Problems with Partial Derivatives",
    "Math Word Problems with Areas, Lengths, and Volumes",
    "Math Word Problems with Sinusoidal Functions",
    "Math Word Problems with Complex Numbers",
    "Math Word Problems with Multiplication",
    "Math Word Problems with Addition",
    "Math Word Problems with Subtraction",
    "Math Word Problems with Division",
    "Math Word Problems with Fractions",
    "Math Word Problems with Remainders",
    "Math Word Problems with Rounding",
    "Math Word Problems with Percentages",
    "Math Word Problems with Ratios",
    "Math Word Problems with Proportions",
    "Math Word Problems with Exponents and Logarithms"
]

themes = {
    "general": general_themes,
    "code": code_themes,
    "math": math_themes
}

seed_data_dict = {
    "instance_generation":{
        "math":"./seed_data/math_seed_data.json",
        "code":"./seed_data/code_seed_data.json",
        "general":"./seed_data/general_seed_data.json",
    },
    "response_generation":{
        "math":"./seed_data/magpie_math_seed_data_10000.json",
        "code":"./seed_data/magpie_code_seed_data_10000.json",
        "general":"./seed_data/magpie_general_seed_data_10000.json",
    },
    "quality_enhancement":{
        "math":"./seed_data/webinstruct_math_seed_data_10000.json",
        "general":"./seed_data/webinstruct_general_seed_data_10000.json",
        "code":"./seed_data/conala_code_seed_data_10000.json"
    }
}

prompt_dict = {
    "instance_generation":{
        "math":"./prompts/instance_generation_math_prompt.txt",
        "code":"./prompts/instance_generation_code_prompt.txt",
        "general":"./prompts/instance_generation_general_prompt.txt"
    },
    "response_generation":{
        "math":"./prompts/response_generation_math_prompt.txt",
        "code":"./prompts/response_generation_code_prompt.txt",
        "general":"./prompts/response_generation_general_prompt.txt"
    },
    "quality_enhancement":{
        "math":"./prompts/quality_enhancement_math_prompt.txt",
        "general":"./prompts/quality_enhancement_general_prompt.txt",
        "code":"./prompts/quality_enhancement_code_prompt.txt"
    }
}

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

def main(args):

    if args.method == 'quality_enhancement':
        placeholder_formats["test_input_trigger"] = "Revised Question:"
        placeholder_formats["test_output_trigger"] = "Revised Answer:"

    with open(seed_data_dict[args.method][args.domain], "r") as f:
        seed_data = json.load(f)

    with open(prompt_dict[args.method][args.domain], "r") as f:
        prompt_template = f.read()
    if args.api_option == "openai":
        llm = OpenAILLM(model_name=args.model_name, api_key=args.api_key)
    elif args.api_option == "litellm":
        llm = LiteLLM(model_name=args.model_name, api_key=args.api_key, api_base=args.api_base, remove_stop=True if "meta-llama" in args.model_name else False)
    # elif args.api_option == "vllm":
    #     llm = VLLM(model_name=args.model_name, api_key=args.api_key, api_base="https://cmu.litellm.ai")

    if args.method == "instance_generation":
        if args.domain == "general":
            placeholder_formats["first_word"] = "<first_word>"
            prompt_loader = ThemeBasedInstanceGenerationPromptLoader(
                prompt_template=prompt_template,
                seed_data=seed_data,
                num_fewshot=3,
                placeholder_formats=placeholder_formats,
                num_sample_from_seed_data=2,
                input_theme_list=themes[args.domain],
                first_word_list=general_triggers
            )
            
        else:
            prompt_loader = ThemeBasedInstanceGenerationPromptLoader(
                prompt_template=prompt_template,
                seed_data=seed_data,
                num_fewshot=3,
                placeholder_formats=placeholder_formats,
                num_sample_from_seed_data=2,
                input_theme_list=themes[args.domain]
            )
        parser = InstanceGenerationParser()

    elif args.method == "response_generation":
        prompt_loader = ResponseGenerationPromptLoader(
            prompt_template=prompt_template,
            seed_data=seed_data,
            placeholder_formats=placeholder_formats
        )
        parser = ResponseGenerationParser()

    else:  # quality_enhancement
        prompt_loader = QualityEnhancementPromptLoader(
            prompt_template=prompt_template,
            seed_data=seed_data,
            placeholder_formats=placeholder_formats
        )
        parser = QualityEnhancementParser()

    if args.parser_style == "json_parser":
        prompt_loader = JSONPromptLoader(
                prompt_template=prompt_template,
                seed_data=seed_data,
                num_fewshot=3,
                placeholder_formats=placeholder_formats,
                num_sample_from_seed_data=2,
            )
        parser = JSONParser()
    else:
        assert args.parser_style == "default_parser"

    if args.domain == "general":
        validator = GeneralValidator(
            tokenizer_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            max_tokens=args.max_tokens,
            placeholder_formats=placeholder_formats
        )
    elif args.domain == "math":
        validator = MathValidator(
            tokenizer_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            max_tokens=args.max_tokens,
            placeholder_formats=placeholder_formats
        )
    else:  # code
        validator = CodeValidator(
            tokenizer_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            max_tokens=args.max_tokens,
            placeholder_formats=placeholder_formats
        )

    sampling_params = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": 0.9,
        "stop": placeholder_formats["stop_phrase"]
    }

    agora_config = AgoraConfig(
        max_retries=10,
        retry_delay=5.0,
        show_progress=True,
        system_message="You are a data generator agent that generates novel instances based on the guidelines, requirements, and examples provided."
    )

    agora = Agora(
        llm=llm,
        placeholder_formats=placeholder_formats,
        prompt_loader=prompt_loader,
        parser=parser,
        validator=validator,
        sampling_params=sampling_params,
        config=agora_config
    )

    output_filename = f"./results/{args.method}_{args.domain}_{args.model_name.replace('/', '_')}.json"
    result = agora.run(num_instances=args.num_instances, num_threads=args.num_threads, output_file=output_filename, cache_file=args.cache_file if args.cache_file is not None else None)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run data augmentation with specific parameters')
    parser.add_argument('--api_option', type=str, required=True, choices=['openai', 'litellm', 'vllm'],
                      help='API option for defining the LLM class')
    parser.add_argument('--method', type=str, required=True, choices=['instance_generation', 'response_generation', 'quality_enhancement'],
                      help='Method for data augmentation')
    parser.add_argument('--domain', type=str, required=True, choices=['general', 'math', 'code'],
                      help='Domain for data augmentation')
    parser.add_argument('--model_name', type=str, required=True,
                      help='Model name for OpenAI LLM')
    parser.add_argument('--api_key', type=str, required=True,
                      help='API key')
    parser.add_argument('--api_base', type=str, required=True,
                      help='API base')
    parser.add_argument('--max_tokens', type=int, default=4096,
                      help='Maximum number of tokens')
    parser.add_argument('--temperature', type=float, default=1.0,
                      help='Temperature for sampling')
    parser.add_argument('--num_instances', type=int, default=1,
                      help='Number of instances to generate')
    parser.add_argument('--num_threads', type=int, default=4,
                      help='Number of threads to use')
    parser.add_argument('--parser_style', type=str, default="default_parser",
                      choices=['default_parser', 'json_parser'],)
    parser.add_argument('--cache_file', type=str, default=None,
                      help='Cache file name for results (default: results_<method>_<domain>_<model_name>.json)')

    args = parser.parse_args()
    main(args)