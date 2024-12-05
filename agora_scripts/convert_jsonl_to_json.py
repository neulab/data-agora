import argparse
import json
from pathlib import Path

def convert_jsonl_to_json(input_file: str):
    results = []
    input_path = Path(input_file)
    output_file = str(input_path).replace('.jsonl', '.json')
    
    with open(input_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))
            
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Convert JSONL file to JSON')
    parser.add_argument('--input_file', type=str, help='Input JSONL file path')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {args.input_file} does not exist")
        
    if not input_path.suffix == '.jsonl':
        raise ValueError("Input file must have .jsonl extension")
        
    convert_jsonl_to_json(args.input_file)

if __name__ == '__main__':
    main()

