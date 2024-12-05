import json
import os
import argparse
from datasets import Dataset
from huggingface_hub import HfApi, HfFolder

def process_json_files(file_path):
    all_data = []
    seen_idx = set()
    with open(file_path, 'r') as f:
        data = json.load(f)
        
        for item in data:
            # if item['idx'] in seen_idx:
            #     continue
            # seen_idx.add(item['idx'])
            
            inst_ = item['instruction']
            resp_ = item['response']
            
            if inst_.startswith("[INPUT]"):
                inst_ = inst_.replace("[INPUT]", "", 1)
            elif inst_.startswith("INPUT: "):
                inst_ = inst_.replace("INPUT: ", "", 1)

            if "code" in file_path:
                if "# Test" in resp_:
                    resp_ = resp_.split("# Test")[0] +"\n```"
                if "// Test" in resp_:
                    resp_ = resp_.split("// Test")[0] + "\n```"
                if "# Example" in resp_:
                    resp_ = resp_.split("# Example")[0] + "\n```"
                if "// Example" in resp_:
                    resp_ = resp_.split("// Example")[0] + "\n```"
                if "# Output" in resp_:
                    resp_ = resp_.split("# Output")[0] + "\n```"
                if "// Output" in resp_:
                    resp_ = resp_.split("// Output")[0] + "\n```"
            
            inst_ = inst_.split("Revised Question:")[-1].split("revised Question:")[-1].split("Revised Question")[-1].split("<revised_question>")[-1].split("Revised Instance:")[-1].split("revised instance:")[-1].split("Revised Instance")[-1].split("Revised Answer:")[0].split("revised Answer:")[0].replace("revised question","question").replace("revised answer","answer").replace("Revised Answer","answer").strip()
            for i in range(20):
                inst_ = inst_.replace(f"### Step {i}:","\n")
                inst_ = inst_.replace(f"## Step {i}:","\n")
                inst_ = inst_.replace(f"# Step {i}:","\n")
            resp_ = resp_.split("Note: ")[0].replace("revised question","question").replace("revised answer","answer").strip()

            if inst_.endswith("\n\n"):
                inst_ = inst_[:-2]

            if resp_.endswith("\n\n"):
                resp_ = resp_[:-2]

            processed_item = {
                "config": item['config'],
                "instruction": inst_,
                "response": resp_
            }
            all_data.append(processed_item)
    return all_data

def upload_to_huggingface(data, dataset_name, hf_key):

    dataset = Dataset.from_list(data)
    
    # Create a DatasetDict with a 'train' split
    from datasets import DatasetDict
    dataset_dict = DatasetDict({"train": dataset})
    
    # Login to Hugging Face (make sure you have set your API token)
    api = HfApi()
    
    # Push the dataset to the Hugging Face Hub as a private dataset
    dataset_dict.push_to_hub(dataset_name, token=hf_key, private=True)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process JSON files and upload to Hugging Face Dataset")
    parser.add_argument("--j", help="Path to the directory containing JSON files")
    parser.add_argument("--d", help="Name of the dataset on Hugging Face (e.g., 'username/dataset-name')")
    parser.add_argument("--hf_key", help="Path to the Hugging Face API key file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    processed_data = process_json_files(args.j)

    upload_to_huggingface(processed_data, args.d, args.hf_key)
    print(f"Dataset uploaded successfully to {args.d}")

    if "50000" in args.j:
        # Upload 10000 samples
        upload_to_huggingface(processed_data[:10000], args.d.replace("50000", "10000"))
        print(f"Dataset with 10000 samples uploaded successfully to {args.d.replace('50000', '10000')}")

        # Upload 25000 samples
        upload_to_huggingface(processed_data[:25000], args.d.replace("50000", "25000"))
        print(f"Dataset with 25000 samples uploaded successfully to {args.d.replace('50000', '25000')}")
