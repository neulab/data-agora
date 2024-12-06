import json

from data_agora import Agora, AgoraConfig
from data_agora.core.llms.local_vllm import LocalVLLM

# from agora.core.llms.test import TestLLM
from data_agora.core.llms.openai import OpenAILLM
from data_agora.core.parsers import InstanceGenerationParser, QualityEnhancementParser, ResponseGenerationParser
from data_agora.core.prompt_loaders import (
    InstanceGenerationPromptLoader,
    PromptResult,
    QualityEnhancementPromptLoader,
    ResponseGenerationPromptLoader,
    ThemeBasedInstanceGenerationPromptLoader,
    load_prompt_loader,
)
from data_agora.core.validators import CodeValidator, GeneralValidator, MathValidator

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
    "Puzzles and Logical Reasoning",
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
    "As a " "Question: ",
    "Express ",
    "Let ",
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
    "Fixing a PHP Code",
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
    "Math Word Problems with Exponents and Logarithms",
]

themes = {"general": general_themes, "code": code_themes, "math": math_themes}

seed_data_dict = {
    "instance_generation": {
        "math": "./tests/seed_data/math_seed_data_subsampled.json",
        "code": "./tests/seed_data/code_seed_data_subsampled.json",
        "general": "./tests/seed_data/general_seed_data_subsampled.json",
    },
    "response_generation": {
        "math": "./tests/seed_data/magpie_math_seed_data_10000_subsampled.json",
        "code": "./tests/seed_data/magpie_code_seed_data_10000_subsampled.json",
        "general": "./tests/seed_data/magpie_general_seed_data_10000_subsampled.json",
    },
    "quality_enhancement": {
        "math": "./tests/seed_data/webinstruct_math_seed_data_10000_subsampled.json",
        "general": "./tests/seed_data/webinstruct_general_seed_data_10000_subsampled.json",
    },
}

prompt_dict = {
    "instance_generation": {
        "math": "./_prompts/math_prompt.txt",
        "code": "./_prompts/code_prompt.txt",
        "general": "./_prompts/general_prompt.txt",
    },
    "response_generation": {
        "math": "./_prompts/magpie_math_prompt.txt",
        "code": "./_prompts/magpie_code_prompt.txt",
        "general": "./_prompts/magpie_general_prompt.txt",
    },
    "quality_enhancement": {
        "math": "./_prompts/webinstruct_math_prompt.txt",
        "general": "./_prompts/webinstruct_general_prompt.txt",
    },
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
    # "first_word": "<first_word>",
}

# llm = OpenAILLM(model_name="gpt-4o-mini-2024-07-18",api_key="")
llm = LocalVLLM(model="meta-llama/Llama-3.1-8B-Instruct")

for method in ["instance_generation", "response_generation", "quality_enhancement"]:
    for domain in ["general", "math", "code"]:
        if method == "quality_enhancement" and domain == "code":
            continue

        if method == "quality_enhancement":
            placeholder_formats["test_input_trigger"] = "Revised Question:"
            placeholder_formats["test_output_trigger"] = "Revised Answer:"

        with open(seed_data_dict[method][domain], "r") as f:
            seed_data = json.load(f)

        with open(prompt_dict[method][domain], "r") as f:
            prompt_template = f.read()

        if method == "instance_generation":
            pass
            # prompt_loader = load_prompt_loader(method, prompt_template=prompt_template, seed_data=seed_data, num_fewshot=3, placeholder_formats=placeholder_formats)
            if domain == "general":
                prompt_loader = ThemeBasedInstanceGenerationPromptLoader(
                    prompt_template=prompt_template,
                    seed_data=seed_data,
                    num_fewshot=3,
                    placeholder_formats=placeholder_formats,
                    num_sample_from_seed_data=2,
                    input_theme_list=themes[domain],
                    first_word_list=general_triggers,
                )
            else:
                prompt_loader = ThemeBasedInstanceGenerationPromptLoader(
                    prompt_template=prompt_template,
                    seed_data=seed_data,
                    num_fewshot=3,
                    placeholder_formats=placeholder_formats,
                    num_sample_from_seed_data=2,
                    input_theme_list=themes[domain],
                )
            parser = InstanceGenerationParser()

        elif method == "response_generation":
            # prompt_loader = load_prompt_loader(method, prompt_template=prompt_template, seed_data=seed_data, placeholder_formats=placeholder_formats)
            prompt_loader = ResponseGenerationPromptLoader(
                prompt_template=prompt_template, seed_data=seed_data, placeholder_formats=placeholder_formats
            )
            parser = ResponseGenerationParser()

        else:
            # prompt_loader = load_prompt_loader(method, prompt_template=prompt_template, seed_data=seed_data, placeholder_formats=placeholder_formats)
            prompt_loader = QualityEnhancementPromptLoader(
                prompt_template=prompt_template, seed_data=seed_data, placeholder_formats=placeholder_formats
            )
            parser = QualityEnhancementParser()

        if domain == "general":
            validator = GeneralValidator(
                tokenizer_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
                max_tokens=4096,
                placeholder_formats=placeholder_formats,
            )
        elif domain == "math":
            validator = MathValidator(
                tokenizer_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
                max_tokens=4096,
                placeholder_formats=placeholder_formats,
            )
        elif domain == "code":
            validator = CodeValidator(
                tokenizer_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
                max_tokens=4096,
                placeholder_formats=placeholder_formats,
            )

        sampling_params = {
            "max_tokens": 4096,
            "temperature": 1.0,
            "top_p": 0.9,
            "stop": placeholder_formats["stop_phrase"],
        }
        agora = Agora(
            llm=llm,
            placeholder_formats=placeholder_formats,
            prompt_loader=prompt_loader,
            parser=parser,
            validator=validator,
            sampling_params=sampling_params,
        )

        result = agora.run(num_instances=1, num_threads=4, output_file="test_output.json")
