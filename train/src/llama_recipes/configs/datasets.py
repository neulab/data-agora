# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    trust_remote_code: bool = False


@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv"
    test_split: str = "src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"


@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_recipes/datasets/alpaca_data.json"

@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "recipes/quickstart/finetuning/datasets/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"
    data_path: str = ""
    
@dataclass
class llamaguard_toxicchat_dataset:
    dataset: str = "llamaguard_toxicchat_dataset"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class math_gpt4o_mini_25000:
    dataset: str = "math_gpt4o_mini_25000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "math_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class math_gpt4o_mini_50000:
    dataset: str = "math_gpt4o_mini_50000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "math_50000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class code_gpt4o_mini_10000:
    dataset: str = "code_gpt4o_mini_10000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "code_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class code_gpt4o_mini_25000:
    dataset: str = "code_gpt4o_mini_25000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "code_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class code_gpt4o_mini_50000:
    dataset: str = "code_gpt4o_mini_50000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "code_50000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class general_gpt4o_mini_10000:
    dataset: str = "general_gpt4o_mini_10000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "general_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class general_gpt4o_mini_25000:
    dataset: str = "general_gpt4o_mini_25000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "general_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class general_gpt4o_mini_50000:
    dataset: str = "general_gpt4o_mini_50000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "general_50000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class math_gpt4o_25000:
    dataset: str = "math_gpt4o_25000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o"
    domain: str = "math_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class code_gpt4o_10000:
    dataset: str = "code_gpt4o_10000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o"
    domain: str = "code_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class general_gpt4o_10000:
    dataset: str = "general_gpt4o_10000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o"
    domain: str = "general_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class math_llama3_70b_25000:
    dataset: str = "math_llama3_70b_25000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "math_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class math_llama3_70b_50000:
    dataset: str = "math_llama3_70b_50000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "math_50000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class code_llama3_70b_10000:
    dataset: str = "code_llama3_70b_10000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "code_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class code_llama3_70b_25000:
    dataset: str = "code_llama3_70b_25000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "code_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class code_llama3_70b_50000:
    dataset: str = "code_llama3_70b_50000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "code_50000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class general_llama3_70b_10000:
    dataset: str = "general_llama3_70b_10000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "general_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class general_llama3_70b_25000:
    dataset: str = "general_llama3_70b_25000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "general_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class general_llama3_70b_50000:
    dataset: str = "general_llama3_70b_50000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "general_50000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class math_llama3_405b_25000:
    dataset: str = "math_llama3_405b_25000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_405b"
    domain: str = "math_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class code_llama3_405b_10000:
    dataset: str = "code_llama3_405b_10000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_405b"
    domain: str = "code_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class general_llama3_405b_10000:
    dataset: str = "general_llama3_405b_10000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_405b"
    domain: str = "general_10000"
    train_split: str = "train"
    test_split: str = "test"


@dataclass
class math_llama3_8b_25000:
    dataset: str = "math_llama3_8b_25000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "math_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class math_llama3_8b_50000:
    dataset: str = "math_llama3_8b_50000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "math_50000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class code_llama3_8b_10000:
    dataset: str = "code_llama3_8b_10000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "code_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class code_llama3_8b_25000:
    dataset: str = "code_llama3_8b_25000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "code_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class code_llama3_8b_50000:
    dataset: str = "code_llama3_8b_50000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "code_50000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class general_llama3_8b_10000:
    dataset: str = "general_llama3_8b_10000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "general_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class general_llama3_8b_25000:
    dataset: str = "general_llama3_8b_25000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "general_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class general_llama3_8b_50000:
    dataset: str = "general_llama3_8b_50000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "general_50000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class math_claude3_sonnet_25000:
    dataset: str = "math_claude3_sonnet_25000"
    trust_remote_code: bool = True
    teacher_model: str = "claude3_sonnet"
    domain: str = "math_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class code_claude3_sonnet_10000:
    dataset: str = "code_claude3_sonnet_10000"
    trust_remote_code: bool = True
    teacher_model: str = "claude3_sonnet"
    domain: str = "code_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class general_claude3_sonnet_10000:
    dataset: str = "general_claude3_sonnet_10000"
    trust_remote_code: bool = True
    teacher_model: str = "claude3_sonnet"
    domain: str = "general_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class general_seed_data:
    dataset: str = "general_seed_data"
    trust_remote_code: bool = True
    teacher_model: str = "human"
    domain: str = "general"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class code_seed_data:
    dataset: str = "code_seed_data"
    trust_remote_code: bool = True
    teacher_model: str = "human"
    domain: str = "code"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class math_seed_data:
    dataset: str = "math_seed_data"
    trust_remote_code: bool = True
    teacher_model: str = "human"
    domain: str = "math"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_math_gpt4o_mini_10000:
    dataset: str = "magpie_math_gpt4o_mini_10000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "magpie_math_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_math_gpt4o_mini_25000:
    dataset: str = "magpie_math_gpt4o_mini_25000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "magpie_math_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_math_gpt4o_mini_50000:
    dataset: str = "magpie_math_gpt4o_mini_50000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "magpie_math_50000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_code_gpt4o_mini_10000:
    dataset: str = "magpie_code_gpt4o_mini_10000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "magpie_code_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_code_gpt4o_mini_25000:
    dataset: str = "magpie_code_gpt4o_mini_25000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "magpie_code_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_code_gpt4o_mini_50000:
    dataset: str = "magpie_code_gpt4o_mini_50000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "magpie_code_50000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_general_gpt4o_mini_10000:
    dataset: str = "magpie_general_gpt4o_mini_10000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "magpie_general_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_general_gpt4o_mini_25000:
    dataset: str = "magpie_general_gpt4o_mini_25000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "magpie_general_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_general_gpt4o_mini_50000:
    dataset: str = "magpie_general_gpt4o_mini_50000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "magpie_general_50000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_math_gpt4o_10000:
    dataset: str = "magpie_math_gpt4o_10000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o"
    domain: str = "magpie_math_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_code_gpt4o_10000:
    dataset: str = "magpie_code_gpt4o_10000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o"
    domain: str = "magpie_code_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_general_gpt4o_10000:
    dataset: str = "magpie_general_gpt4o_10000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o"
    domain: str = "magpie_general_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_math_llama3_70b_10000:
    dataset: str = "magpie_math_llama3_70b_10000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "magpie_math_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_math_llama3_70b_25000:
    dataset: str = "magpie_math_llama3_70b_25000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "magpie_math_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_math_llama3_70b_50000:
    dataset: str = "magpie_math_llama3_70b_50000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "magpie_math_50000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_code_llama3_70b_10000:
    dataset: str = "magpie_code_llama3_70b_10000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "magpie_code_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_code_llama3_70b_25000:
    dataset: str = "magpie_code_llama3_70b_25000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "magpie_code_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_code_llama3_70b_50000:
    dataset: str = "magpie_code_llama3_70b_50000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "magpie_code_50000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_general_llama3_70b_10000:
    dataset: str = "magpie_general_llama3_70b_10000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "magpie_general_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_general_llama3_70b_25000:
    dataset: str = "magpie_general_llama3_70b_25000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "magpie_general_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_general_llama3_70b_50000:
    dataset: str = "magpie_general_llama3_70b_50000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "magpie_general_50000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_math_llama3_405b_10000:
    dataset: str = "magpie_math_llama3_405b_10000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_405b"
    domain: str = "magpie_math_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_code_llama3_405b_10000:
    dataset: str = "magpie_code_llama3_405b_10000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_405b"
    domain: str = "magpie_code_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_general_llama3_405b_10000:
    dataset: str = "magpie_general_llama3_405b_10000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_405b"
    domain: str = "magpie_general_10000"
    train_split: str = "train"
    test_split: str = "test"


@dataclass
class magpie_math_llama3_8b_10000:
    dataset: str = "magpie_math_llama3_8b_10000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "magpie_math_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_math_llama3_8b_25000:
    dataset: str = "magpie_math_llama3_8b_25000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "magpie_math_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_math_llama3_8b_50000:
    dataset: str = "magpie_math_llama3_8b_50000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "magpie_math_50000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_code_llama3_8b_10000:
    dataset: str = "magpie_code_llama3_8b_10000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "magpie_code_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_code_llama3_8b_25000:
    dataset: str = "magpie_code_llama3_8b_25000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "magpie_code_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_code_llama3_8b_50000:
    dataset: str = "magpie_code_llama3_8b_50000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "magpie_code_50000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_general_llama3_8b_10000:
    dataset: str = "magpie_general_llama3_8b_10000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "magpie_general_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_general_llama3_8b_25000:
    dataset: str = "magpie_general_llama3_8b_25000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "magpie_general_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_general_llama3_8b_50000:
    dataset: str = "magpie_general_llama3_8b_50000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "magpie_general_50000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_math_claude3_sonnet_10000:
    dataset: str = "magpie_math_claude3_sonnet_10000"
    trust_remote_code: bool = True
    teacher_model: str = "claude3_sonnet"
    domain: str = "magpie_math_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_code_claude3_sonnet_10000:
    dataset: str = "magpie_code_claude3_sonnet_10000"
    trust_remote_code: bool = True
    teacher_model: str = "claude3_sonnet"
    domain: str = "magpie_code_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class magpie_general_claude3_sonnet_10000:
    dataset: str = "magpie_general_claude3_sonnet_10000"
    trust_remote_code: bool = True
    teacher_model: str = "claude3_sonnet"
    domain: str = "magpie_general_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_math_gpt4o_mini_10000:
    dataset: str = "webinstruct_math_gpt4o_mini_10000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "webinstruct_math_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_math_gpt4o_mini_25000:
    dataset: str = "webinstruct_math_gpt4o_mini_25000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "webinstruct_math_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_math_gpt4o_mini_50000:
    dataset: str = "webinstruct_math_gpt4o_mini_50000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "webinstruct_math_50000"
    train_split: str = "train"
    test_split: str = "test"


@dataclass
class webinstruct_general_gpt4o_mini_10000:
    dataset: str = "webinstruct_general_gpt4o_mini_10000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "webinstruct_general_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_general_gpt4o_mini_25000:
    dataset: str = "webinstruct_general_gpt4o_mini_25000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "webinstruct_general_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_general_gpt4o_mini_50000:
    dataset: str = "webinstruct_general_gpt4o_mini_50000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "webinstruct_general_50000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_code_gpt4o_mini_10000:
    dataset: str = "webinstruct_code_gpt4o_mini_10000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "webinstruct_code_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_code_gpt4o_mini_25000:
    dataset: str = "webinstruct_code_gpt4o_mini_25000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "webinstruct_code_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_code_gpt4o_mini_50000:
    dataset: str = "webinstruct_code_gpt4o_mini_50000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "webinstruct_code_50000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_math_gpt4o_10000:
    dataset: str = "webinstruct_math_gpt4o_10000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o"
    domain: str = "webinstruct_math_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_general_gpt4o_10000:
    dataset: str = "webinstruct_general_gpt4o_10000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o"
    domain: str = "webinstruct_general_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_code_gpt4o_10000:
    dataset: str = "webinstruct_code_gpt4o_10000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o"
    domain: str = "webinstruct_code_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_math_llama3_70b_10000:
    dataset: str = "webinstruct_math_llama3_70b_10000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "webinstruct_math_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_math_llama3_70b_25000:
    dataset: str = "webinstruct_math_llama3_70b_25000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "webinstruct_math_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_math_llama3_70b_50000:
    dataset: str = "webinstruct_math_llama3_70b_50000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "webinstruct_math_50000"
    train_split: str = "train"
    test_split: str = "test"


@dataclass
class webinstruct_general_llama3_70b_10000:
    dataset: str = "webinstruct_general_llama3_70b_10000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "webinstruct_general_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_general_llama3_70b_25000:
    dataset: str = "webinstruct_general_llama3_70b_25000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "webinstruct_general_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_general_llama3_70b_50000:
    dataset: str = "webinstruct_general_llama3_70b_50000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "webinstruct_general_50000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_code_llama3_70b_10000:
    dataset: str = "webinstruct_code_llama3_70b_10000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "webinstruct_code_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_code_llama3_70b_25000:
    dataset: str = "webinstruct_code_llama3_70b_25000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "webinstruct_code_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_code_llama3_70b_50000:
    dataset: str = "webinstruct_code_llama3_70b_50000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "webinstruct_code_50000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_math_llama3_405b_10000:
    dataset: str = "webinstruct_math_llama3_405b_10000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_405b"
    domain: str = "webinstruct_math_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_general_llama3_405b_10000:
    dataset: str = "webinstruct_general_llama3_405b_10000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_405b"
    domain: str = "webinstruct_general_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_code_llama3_405b_10000:
    dataset: str = "webinstruct_code_llama3_405b_10000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_405b"
    domain: str = "webinstruct_code_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_math_llama3_8b_10000:
    dataset: str = "webinstruct_math_llama3_8b_10000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "webinstruct_math_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_math_llama3_8b_25000:
    dataset: str = "webinstruct_math_llama3_8b_25000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "webinstruct_math_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_math_llama3_8b_50000:
    dataset: str = "webinstruct_math_llama3_8b_50000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "webinstruct_math_50000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_general_llama3_8b_10000:
    dataset: str = "webinstruct_general_llama3_8b_10000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "webinstruct_general_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_general_llama3_8b_25000:
    dataset: str = "webinstruct_general_llama3_8b_25000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "webinstruct_general_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_general_llama3_8b_50000:
    dataset: str = "webinstruct_general_llama3_8b_50000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "webinstruct_general_50000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_code_llama3_8b_10000:
    dataset: str = "webinstruct_code_llama3_8b_10000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "webinstruct_code_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_code_llama3_8b_25000:
    dataset: str = "webinstruct_code_llama3_8b_25000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "webinstruct_code_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_code_llama3_8b_50000:
    dataset: str = "webinstruct_code_llama3_8b_50000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "webinstruct_code_50000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_math_claude3_sonnet_10000:
    dataset: str = "webinstruct_math_claude3_sonnet_10000"
    trust_remote_code: bool = True
    teacher_model: str = "claude3_sonnet"
    domain: str = "webinstruct_math_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_general_claude3_sonnet_10000:
    dataset: str = "webinstruct_general_claude3_sonnet_10000"
    trust_remote_code: bool = True
    teacher_model: str = "claude3_sonnet"
    domain: str = "webinstruct_general_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class webinstruct_code_claude3_sonnet_10000:
    dataset: str = "webinstruct_code_claude3_sonnet_10000"
    trust_remote_code: bool = True
    teacher_model: str = "claude3_sonnet"
    domain: str = "webinstruct_code_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class fineweb_edu_general_gpt4o_mini_10000:
    dataset: str = "fineweb_edu_general_gpt4o_mini_10000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "fineweb_edu_general_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class fineweb_edu_general_gpt4o_mini_25000:
    dataset: str = "fineweb_edu_general_gpt4o_mini_25000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "fineweb_edu_general_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class fineweb_edu_general_gpt4o_mini_50000:
    dataset: str = "fineweb_edu_general_gpt4o_mini_50000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "fineweb_edu_general_50000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class fineweb_edu_general_gpt4o_10000:
    dataset: str = "fineweb_edu_general_gpt4o_10000"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o"
    domain: str = "fineweb_edu_general_10000"
    train_split: str = "train"
    test_split: str = "test"


@dataclass
class fineweb_edu_general_llama3_70b_10000:
    dataset: str = "fineweb_edu_general_llama3_70b_10000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "fineweb_edu_general_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class fineweb_edu_general_llama3_70b_25000:
    dataset: str = "fineweb_edu_general_llama3_70b_25000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "fineweb_edu_general_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class fineweb_edu_general_llama3_70b_50000:
    dataset: str = "fineweb_edu_general_llama3_70b_50000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "fineweb_edu_general_50000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class fineweb_edu_general_llama3_405b_10000:
    dataset: str = "fineweb_edu_general_llama3_405b_10000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_405b"
    domain: str = "fineweb_edu_general_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class fineweb_edu_general_llama3_8b_10000:
    dataset: str = "fineweb_edu_general_llama3_8b_10000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "fineweb_edu_general_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class fineweb_edu_general_llama3_8b_25000:
    dataset: str = "fineweb_edu_general_llama3_8b_25000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "fineweb_edu_general_25000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class fineweb_edu_general_llama3_8b_50000:
    dataset: str = "fineweb_edu_general_llama3_8b_50000"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "fineweb_edu_general_50000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class fineweb_edu_general_claude3_sonnet_10000:
    dataset: str = "fineweb_edu_general_claude3_sonnet_10000"
    trust_remote_code: bool = True
    teacher_model: str = "claude3_sonnet"
    domain: str = "fineweb_edu_general_10000"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class ablation1_math_llama3_8b_instruct:
    dataset: str = "ablation1_math_llama3_8b_instruct"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "ablation1_math"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class ablation1_math_llama3_70b_instruct:
    dataset: str = "ablation1_math_llama3_70b_instruct"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "ablation1_math"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class ablation1_math_gpt4o_mini:
    dataset: str = "ablation1_math_gpt4o_mini"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "ablation1_math"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class ablation1_code_llama3_8b_instruct:
    dataset: str = "ablation1_code_llama3_8b_instruct"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "ablation1_code"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class ablation1_code_llama3_70b_instruct:
    dataset: str = "ablation1_code_llama3_70b_instruct"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "ablation1_code"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class ablation1_code_gpt4o_mini:
    dataset: str = "ablation1_code_gpt4o_mini"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "ablation1_code"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class ablation1_general_llama3_8b_instruct:
    dataset: str = "ablation1_general_llama3_8b_instruct"
    trust_remote_general: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "ablation1_general"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class ablation1_general_llama3_70b_instruct:
    dataset: str = "ablation1_general_llama3_70b_instruct"
    trust_remote_general: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "ablation1_general"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class ablation1_general_gpt4o_mini:
    dataset: str = "ablation1_general_gpt4o_mini"
    trust_remote_general: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "ablation1_general"
    train_split: str = "train"
    test_split: str = "test"



@dataclass
class ablation2_math_llama3_8b_instruct:
    dataset: str = "ablation2_math_llama3_8b_instruct"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "ablation2_math"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class ablation2_math_llama3_70b_instruct:
    dataset: str = "ablation2_math_llama3_70b_instruct"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "ablation2_math"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class ablation2_math_gpt4o_mini:
    dataset: str = "ablation2_math_gpt4o_mini"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "ablation2_math"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class ablation2_code_llama3_8b_instruct:
    dataset: str = "ablation2_code_llama3_8b_instruct"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "ablation2_code"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class ablation2_code_llama3_70b_instruct:
    dataset: str = "ablation2_code_llama3_70b_instruct"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "ablation2_code"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class ablation2_code_gpt4o_mini:
    dataset: str = "ablation2_code_gpt4o_mini"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "ablation2_code"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class ablation2_general_llama3_8b_instruct:
    dataset: str = "ablation2_general_llama3_8b_instruct"
    trust_remote_general: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "ablation2_general"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class ablation2_general_llama3_70b_instruct:
    dataset: str = "ablation2_general_llama3_70b_instruct"
    trust_remote_general: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "ablation2_general"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class ablation2_general_gpt4o_mini:
    dataset: str = "ablation2_general_gpt4o_mini"
    trust_remote_general: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "ablation2_general"
    train_split: str = "train"
    test_split: str = "test"


@dataclass
class ablation3_math_llama3_8b_instruct:
    dataset: str = "ablation3_math_llama3_8b_instruct"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "ablation3_math"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class ablation3_math_llama3_70b_instruct:
    dataset: str = "ablation3_math_llama3_70b_instruct"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "ablation3_math"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class ablation3_math_gpt4o_mini:
    dataset: str = "ablation3_math_gpt4o_mini"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "ablation3_math"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class ablation3_code_llama3_8b_instruct:
    dataset: str = "ablation3_code_llama3_8b_instruct"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "ablation3_code"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class ablation3_code_llama3_70b_instruct:
    dataset: str = "ablation3_code_llama3_70b_instruct"
    trust_remote_code: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "ablation3_code"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class ablation3_code_gpt4o_mini:
    dataset: str = "ablation3_code_gpt4o_mini"
    trust_remote_code: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "ablation3_code"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class ablation3_general_llama3_8b_instruct:
    dataset: str = "ablation3_general_llama3_8b_instruct"
    trust_remote_general: bool = True
    teacher_model: str = "llama3_8b"
    domain: str = "ablation3_general"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class ablation3_general_llama3_70b_instruct:
    dataset: str = "ablation3_general_llama3_70b_instruct"
    trust_remote_general: bool = True
    teacher_model: str = "llama3_70b"
    domain: str = "ablation3_general"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class ablation3_general_gpt4o_mini:
    dataset: str = "ablation3_general_gpt4o_mini"
    trust_remote_general: bool = True
    teacher_model: str = "gpt4o-mini"
    domain: str = "ablation3_general"
    train_split: str = "train"
    test_split: str = "test"