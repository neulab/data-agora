import copy
import datasets
import json

import torch
from torch.utils.data import Dataset

data_mapping={
    "gpt4o-mini":{
        "math_25000": "Augmentation-Scaling-Laws/math_gpt4o_mini_25000",
        "math_50000": "Augmentation-Scaling-Laws/math_gpt4o_mini_50000",
        "general_10000": "Augmentation-Scaling-Laws/general_gpt4o_mini_10000",
        "general_25000": "Augmentation-Scaling-Laws/general_gpt4o_mini_25000",
        "general_50000": "Augmentation-Scaling-Laws/general_gpt4o_mini_50000",
        "code_10000": "Augmentation-Scaling-Laws/code_gpt4o_mini_10000",
        "code_25000": "Augmentation-Scaling-Laws/code_gpt4o_mini_25000",
        "code_50000": "Augmentation-Scaling-Laws/code_gpt4o_mini_50000",
        "magpie_math_10000": "Augmentation-Scaling-Laws/magpie_math_gpt4o_mini_10000",
        "magpie_math_25000": "Augmentation-Scaling-Laws/magpie_math_gpt4o_mini_25000",
        "magpie_math_50000": "Augmentation-Scaling-Laws/magpie_math_gpt4o_mini_50000",
        "magpie_general_10000": "Augmentation-Scaling-Laws/magpie_general_gpt4o_mini_10000",
        "magpie_general_25000": "Augmentation-Scaling-Laws/magpie_general_gpt4o_mini_25000",
        "magpie_general_50000": "Augmentation-Scaling-Laws/magpie_general_gpt4o_mini_50000",
        "magpie_code_10000": "Augmentation-Scaling-Laws/magpie_code_gpt4o_mini_10000",
        "magpie_code_25000": "Augmentation-Scaling-Laws/magpie_code_gpt4o_mini_25000",
        "magpie_code_50000": "Augmentation-Scaling-Laws/magpie_code_gpt4o_mini_50000",
        "webinstruct_math_10000": "Augmentation-Scaling-Laws/webinstruct_math_gpt4o_mini_10000",
        "webinstruct_math_25000": "Augmentation-Scaling-Laws/webinstruct_math_gpt4o_mini_25000",
        "webinstruct_math_50000": "Augmentation-Scaling-Laws/webinstruct_math_gpt4o_mini_50000",
        "webinstruct_general_10000": "Augmentation-Scaling-Laws/webinstruct_general_gpt4o_mini_10000",
        "webinstruct_general_25000": "Augmentation-Scaling-Laws/webinstruct_general_gpt4o_mini_25000",
        "webinstruct_general_50000": "Augmentation-Scaling-Laws/webinstruct_general_gpt4o_mini_50000",
        "webinstruct_code_10000": "Augmentation-Scaling-Laws/webinstruct_code_gpt4o_mini_10000",
        "webinstruct_code_25000": "Augmentation-Scaling-Laws/webinstruct_code_gpt4o_mini_25000",
        "webinstruct_code_50000": "Augmentation-Scaling-Laws/webinstruct_code_gpt4o_mini_50000",
        "fineweb_edu_general_10000": "Augmentation-Scaling-Laws/fineweb_edu_general_gpt4o_mini_10000",
        "fineweb_edu_general_25000": "Augmentation-Scaling-Laws/fineweb_edu_general_gpt4o_mini_25000",
        "fineweb_edu_general_50000": "Augmentation-Scaling-Laws/fineweb_edu_general_gpt4o_mini_50000",
        "ablation1_math": "seungone/ablation1_math_gpt4o_mini",
        "ablation1_general": "seungone/ablation1_general_gpt4o_mini",
        "ablation1_code": "seungone/ablation1_code_gpt4o_mini",
        "ablation2_math": "seungone/ablation2_math_gpt4o_mini",
        "ablation2_general": "seungone/ablation2_general_gpt4o_mini",
        "ablation2_code": "seungone/ablation2_code_gpt4o_mini",
        "ablation3_math": "seungone/ablation3_math_gpt4o_mini",
        "ablation3_general": "seungone/ablation3_general_gpt4o_mini",
        "ablation3_code": "seungone/ablation3_code_gpt4o_mini"
    },
    "gpt4o":{
        "math_25000": "Augmentation-Scaling-Laws/math_gpt4o_25000",
        "general_10000": "Augmentation-Scaling-Laws/general_gpt4o_10000",
        "code_10000": "Augmentation-Scaling-Laws/code_gpt4o_10000",
        "magpie_math_10000": "Augmentation-Scaling-Laws/magpie_math_gpt4o_10000",
        "magpie_general_10000": "Augmentation-Scaling-Laws/magpie_general_gpt4o_10000",
        "magpie_code_10000": "Augmentation-Scaling-Laws/magpie_code_gpt4o_10000",
        "webinstruct_math_10000": "Augmentation-Scaling-Laws/webinstruct_math_gpt4o_10000",
        "webinstruct_general_10000": "Augmentation-Scaling-Laws/webinstruct_general_gpt4o_10000",
        "webinstruct_code_10000": "Augmentation-Scaling-Laws/webinstruct_code_gpt4o_10000",
        "fineweb_edu_general_10000": "Augmentation-Scaling-Laws/fineweb_edu_general_gpt4o_10000",
    },
    "llama3_70b":{
        "math_25000": "Augmentation-Scaling-Laws/math_llama3.1_70b_instruct_25000",
        "math_50000": "Augmentation-Scaling-Laws/math_llama3.1_70b_instruct_50000",
        "general_10000": "Augmentation-Scaling-Laws/general_llama3.1_70b_instruct_10000",
        "general_25000": "Augmentation-Scaling-Laws/general_llama3.1_70b_instruct_25000",
        "general_50000": "Augmentation-Scaling-Laws/general_llama3.1_70b_instruct_50000",
        "code_10000": "Augmentation-Scaling-Laws/code_llama3.1_70b_instruct_10000",
        "code_25000": "Augmentation-Scaling-Laws/code_llama3.1_70b_instruct_25000",
        "code_50000": "Augmentation-Scaling-Laws/code_llama3.1_70b_instruct_50000",
        "magpie_math_10000": "Augmentation-Scaling-Laws/magpie_math_llama3.1_70b_instruct_10000",
        "magpie_math_25000": "Augmentation-Scaling-Laws/magpie_math_llama3.1_70b_instruct_25000",
        "magpie_math_50000": "Augmentation-Scaling-Laws/magpie_math_llama3.1_70b_instruct_50000",
        "magpie_general_10000": "Augmentation-Scaling-Laws/magpie_general_llama3.1_70b_instruct_10000",
        "magpie_general_25000": "Augmentation-Scaling-Laws/magpie_general_llama3.1_70b_instruct_25000",
        "magpie_general_50000": "Augmentation-Scaling-Laws/magpie_general_llama3.1_70b_instruct_50000",
        "magpie_code_10000": "Augmentation-Scaling-Laws/magpie_code_llama3.1_70b_instruct_10000",
        "magpie_code_25000": "Augmentation-Scaling-Laws/magpie_code_llama3.1_70b_instruct_25000",
        "magpie_code_50000": "Augmentation-Scaling-Laws/magpie_code_llama3.1_70b_instruct_50000",
        "webinstruct_math_10000": "Augmentation-Scaling-Laws/webinstruct_math_llama3.1_70b_instruct_10000",
        "webinstruct_math_25000": "Augmentation-Scaling-Laws/webinstruct_math_llama3.1_70b_instruct_25000",
        "webinstruct_math_50000": "Augmentation-Scaling-Laws/webinstruct_math_llama3.1_70b_instruct_50000",
        "webinstruct_general_10000": "Augmentation-Scaling-Laws/webinstruct_general_llama3.1_70b_instruct_10000",
        "webinstruct_general_25000": "Augmentation-Scaling-Laws/webinstruct_general_llama3.1_70b_instruct_25000",
        "webinstruct_general_50000": "Augmentation-Scaling-Laws/webinstruct_general_llama3.1_70b_instruct_50000",
        "webinstruct_code_10000": "Augmentation-Scaling-Laws/webinstruct_code_llama3.1_70b_instruct_10000",
        "webinstruct_code_25000": "Augmentation-Scaling-Laws/webinstruct_code_llama3.1_70b_instruct_25000",
        "webinstruct_code_50000": "Augmentation-Scaling-Laws/webinstruct_code_llama3.1_70b_instruct_50000",
        "fineweb_edu_general_10000": "Augmentation-Scaling-Laws/fineweb_edu_general_llama3.1_70b_instruct_10000",
        "fineweb_edu_general_25000": "Augmentation-Scaling-Laws/fineweb_edu_general_llama3.1_70b_instruct_25000",
        "fineweb_edu_general_50000": "Augmentation-Scaling-Laws/fineweb_edu_general_llama3.1_70b_instruct_50000",
        "ablation1_math": "seungone/ablation1_math_llama3.1_70b_instruct",
        "ablation1_general": "seungone/ablation1_general_llama3.1_70b_instruct",
        "ablation1_code": "seungone/ablation1_code_llama3.1_70b_instruct",
        "ablation2_math": "seungone/ablation2_math_llama3.1_70b_instruct",
        "ablation2_general": "seungone/ablation2_general_llama3.1_70b_instruct",
        "ablation2_code": "seungone/ablation2_code_llama3.1_70b_instruct",
        "ablation3_math": "seungone/ablation3_math_llama3.1_70b_instruct",
        "ablation3_general": "seungone/ablation3_general_llama3.1_70b_instruct",
        "ablation3_code": "seungone/ablation3_code_llama3.1_70b_instruct"
    },
    "llama3_405b":{
        "math_25000": "Augmentation-Scaling-Laws/math_llama3.1_405b_instruct_25000",
        "general_10000": "Augmentation-Scaling-Laws/general_llama3.1_405b_instruct_10000",
        "code_10000": "Augmentation-Scaling-Laws/code_llama3.1_405b_instruct_10000",
        "magpie_math_10000": "Augmentation-Scaling-Laws/magpie_math_llama3.1_405b_instruct_10000",
        "magpie_general_10000": "Augmentation-Scaling-Laws/magpie_general_llama3.1_405b_instruct_10000",
        "magpie_code_10000": "Augmentation-Scaling-Laws/magpie_code_llama3.1_405b_instruct_10000",
        "webinstruct_math_10000": "Augmentation-Scaling-Laws/webinstruct_math_llama3.1_405b_instruct_10000",
        "webinstruct_general_10000": "Augmentation-Scaling-Laws/webinstruct_general_llama3.1_405b_instruct_10000",
        "webinstruct_code_10000": "Augmentation-Scaling-Laws/webinstruct_code_llama3.1_405b_instruct_10000",
        "fineweb_edu_general_10000": "Augmentation-Scaling-Laws/fineweb_edu_general_llama3.1_405b_instruct_10000",
    },
    "llama3_8b":{
        "math_25000": "Augmentation-Scaling-Laws/math_llama3.1_8b_instruct_25000",
        "math_50000": "Augmentation-Scaling-Laws/math_llama3.1_8b_instruct_50000",
        "general_10000": "Augmentation-Scaling-Laws/general_llama3.1_8b_instruct_10000",
        "general_25000": "Augmentation-Scaling-Laws/general_llama3.1_8b_instruct_25000",
        "general_50000": "Augmentation-Scaling-Laws/general_llama3.1_8b_instruct_50000",
        "code_10000": "Augmentation-Scaling-Laws/code_llama3.1_8b_instruct_10000",
        "code_25000": "Augmentation-Scaling-Laws/code_llama3.1_8b_instruct_25000",
        "code_50000": "Augmentation-Scaling-Laws/code_llama3.1_8b_instruct_50000",
        "magpie_math_10000": "Augmentation-Scaling-Laws/magpie_math_llama3.1_8b_instruct_10000",
        "magpie_math_25000": "Augmentation-Scaling-Laws/magpie_math_llama3.1_8b_instruct_25000",
        "magpie_math_50000": "Augmentation-Scaling-Laws/magpie_math_llama3.1_8b_instruct_50000",
        "magpie_general_10000": "Augmentation-Scaling-Laws/magpie_general_llama3.1_8b_instruct_10000",
        "magpie_general_25000": "Augmentation-Scaling-Laws/magpie_general_llama3.1_8b_instruct_25000",
        "magpie_general_50000": "Augmentation-Scaling-Laws/magpie_general_llama3.1_8b_instruct_50000",
        "magpie_code_10000": "Augmentation-Scaling-Laws/magpie_code_llama3.1_8b_instruct_10000",
        "magpie_code_25000": "Augmentation-Scaling-Laws/magpie_code_llama3.1_8b_instruct_25000",
        "magpie_code_50000": "Augmentation-Scaling-Laws/magpie_code_llama3.1_8b_instruct_50000",
        "webinstruct_math_10000": "Augmentation-Scaling-Laws/webinstruct_math_llama3.1_8b_instruct_10000",
        "webinstruct_math_25000": "Augmentation-Scaling-Laws/webinstruct_math_llama3.1_8b_instruct_25000",
        "webinstruct_math_50000": "Augmentation-Scaling-Laws/webinstruct_math_llama3.1_8b_instruct_50000",
        "webinstruct_general_10000": "Augmentation-Scaling-Laws/webinstruct_general_llama3.1_8b_instruct_10000",
        "webinstruct_general_25000": "Augmentation-Scaling-Laws/webinstruct_general_llama3.1_8b_instruct_25000",
        "webinstruct_general_50000": "Augmentation-Scaling-Laws/webinstruct_general_llama3.1_8b_instruct_50000",
        "webinstruct_code_10000": "Augmentation-Scaling-Laws/webinstruct_code_llama3.1_8b_instruct_10000",
        "webinstruct_code_25000": "Augmentation-Scaling-Laws/webinstruct_code_llama3.1_8b_instruct_25000",
        "webinstruct_code_50000": "Augmentation-Scaling-Laws/webinstruct_code_llama3.1_8b_instruct_50000",
        "fineweb_edu_general_10000": "Augmentation-Scaling-Laws/fineweb_edu_general_llama3.1_8b_instruct_10000",
        "fineweb_edu_general_25000": "Augmentation-Scaling-Laws/fineweb_edu_general_llama3.1_8b_instruct_25000",
        "fineweb_edu_general_50000": "Augmentation-Scaling-Laws/fineweb_edu_general_llama3.1_8b_instruct_50000",
        "ablation1_math": "seungone/ablation1_math_llama3.1_8b_instruct",
        "ablation1_general": "seungone/ablation1_general_llama3.1_8b_instruct",
        "ablation1_code": "seungone/ablation1_code_llama3.1_8b_instruct",
        "ablation2_math": "seungone/ablation2_math_llama3.1_8b_instruct",
        "ablation2_general": "seungone/ablation2_general_llama3.1_8b_instruct",
        "ablation2_code": "seungone/ablation2_code_llama3.1_8b_instruct",
        "ablation3_math": "seungone/ablation3_math_llama3.1_8b_instruct",
        "ablation3_general": "seungone/ablation3_general_llama3.1_8b_instruct",
        "ablation3_code": "seungone/ablation3_code_llama3.1_8b_instruct",
    },
    "claude3_sonnet":{
        "math_25000": "Augmentation-Scaling-Laws/math_claude3.5_sonnet_25000",
        "general_10000": "Augmentation-Scaling-Laws/general_claude3.5_sonnet_10000",
        "code_10000": "Augmentation-Scaling-Laws/code_claude3.5_sonnet_10000",
        "magpie_math_10000": "Augmentation-Scaling-Laws/magpie_math_claude3.5_sonnet_10000",
        "magpie_general_10000": "Augmentation-Scaling-Laws/magpie_general_claude3.5_sonnet_10000",
        "magpie_code_10000": "Augmentation-Scaling-Laws/magpie_code_claude3.5_sonnet_10000",
        "webinstruct_math_10000": "Augmentation-Scaling-Laws/webinstruct_math_claude3.5_sonnet_10000",
        "webinstruct_general_10000": "Augmentation-Scaling-Laws/webinstruct_general_claude3.5_sonnet_10000",
        "webinstruct_code_10000": "Augmentation-Scaling-Laws/webinstruct_code_claude3.5_sonnet_10000",
        "fineweb_edu_general_10000": "Augmentation-Scaling-Laws/fineweb_edu_general_claude3.5_sonnet_10000",
    },
    "human":{
        "math": "Augmentation-Scaling-Laws/math-seed-data",
        "general": "Augmentation-Scaling-Laws/general-seed-data",
        "code": "Augmentation-Scaling-Laws/code-seed-data"
    }
}


def create_prompt_with_llama3_format(prompt):
    system_message = "You are a helpful AI assistant."
    # <|begin_of_text|> is appended by default below, so no need to do it here
    formatted_text = f"<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>"
    formatted_text += "<|start_header_id|>user<|end_header_id|>\n\n" + prompt + "<|eot_id|>"
    formatted_text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return formatted_text

def create_response_with_llama3_format(response):
    formatted_text = f"{response}<|eot_id|>"
    return formatted_text



class SyntheticDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, train_split):
        self.ann = datasets.load_dataset(data_mapping[dataset_config.teacher_model][dataset_config.domain], split="train", trust_remote_code=True)
        if train_split!="train":
            self.ann = self.ann.filter(lambda example, indice: indice < 500, with_indices=True)
            
        self.tokenizer = tokenizer
        self.tokenizer.pad_token_id = 128001
        self.tokenizer.eos_token_id = 128009
        self.tokenizer.pad_token = "<|end_of_text|>"
        self.tokenizer.eos_token = "<|eot_id|>"

        # self.max_words=4096

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        ann = self.ann[index]

        prompt = create_prompt_with_llama3_format(ann["instruction"])
        response = ann["response"]
        example = prompt + response

        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        # padding = self.max_words - example.shape[0]
        # if padding > 0:
        #     example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        # elif padding < 0:
        #     example = example[: self.max_words]

        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        # if index==len(self.ann)-1:
        #     print("#"*30)
        #     print(self.tokenizer.decode(example.tolist(),skip_special_tokens=False,clean_up_tokenization_spaces=False))
        #     print("@"*30)
        #     print(example.tolist())
        #     print("@"*30)
        #     print(labels.tolist())
        #     print("@"*30)
        #     print(example_mask.tolist())
        #     print("#"*30)

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask":example_mask.tolist(),
        }
