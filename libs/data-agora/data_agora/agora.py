import json
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Union

from tqdm import tqdm

from .core.llms.base import LLM
from .core.parsers import Parser
from .core.prompt_loaders import (
    BasePromptLoader,
    InstanceGenerationPromptLoader,
    ThemeBasedInstanceGenerationPromptLoader,
)
from .core.validators import Validator, validate_keywords


# TODO: If prompt_loader is InstanceGenerationPromptLoader, then there should be option to update the seed_data with the generated instance
@dataclass
class AgoraConfig:
    """Configuration for the Agora process"""

    max_retries: int = 10
    retry_delay: float = 5.0
    show_progress: bool = True
    system_message: str = (
        "You are a data generator agent that generates novel instances based on the guidelines, requirements, and examples provided."
    )


# TODO: Support parallel generation with multiprocessing or threading
class Agora:
    """Orchestrates the data synthesis process using LLM, PromptLoader, and Validator"""

    def __init__(
        self,
        llm: LLM,
        placeholder_formats: Dict[str, str],  # TODO: Shouldn't placeholder_formats be a part of prompt_loader?
        prompt_loader: BasePromptLoader,
        validator: Validator,
        parser: Parser,
        sampling_params: Optional[Dict] = None,
        config: Optional[AgoraConfig] = None,
        verbose: bool = False,
    ):
        """Initialize Agora with components and configuration

        Args:
            llm: Language model interface
            loader: Prompt preparation component
            validator: Output validation component
            sampling_params: Parameters for LLM sampling
            config: Optional configuration settings
        """

        self.llm = llm
        self.placeholder_formats = placeholder_formats

        if verbose:
            print("#" * 30)
            print("Current placeholder formats:", self.placeholder_formats)
            print("#" * 30)

        if "stop_phrase" in self.placeholder_formats:
            self.stop_phrase = self.placeholder_formats["stop_phrase"]
        else:
            self.stop_phrase = None

        self.prompt_loader = prompt_loader
        self.validator = validator
        self.parser = parser
        self.sampling_params = sampling_params or {}
        self.config = config or AgoraConfig()

        if "stop_phrase" not in self.placeholder_formats:
            warnings.warn("'stop_phrase' parameter not specified in placeholder_formats")

        # Add a lock for thread-safe seed data updates
        self._seed_data_lock = Lock()

    def validate_finish_reason(self, completion) -> bool:
        """Validates the finish reason from API response"""

        flg = False
        try:
            try:
                finish_reason = completion["choices"][0]["finish_reason"]
            except:
                try:
                    finish_reason = completion.choices[0].finish_reason
                except:
                    raise ValueError("Invalid finish reason format")
            if finish_reason in ["stop", "stop_sequence"]:
                flg = True
        except:
            flg = False

        if not flg:
            if self.stop_phrase is not None:
                flg = validate_keywords(
                    completion["choices"][0]["message"]["content"], end_keywords=[self.stop_phrase]
                )

        return flg

    def run_single(self, index: Optional[int] = 0) -> Optional[Dict]:
        """Generate a single instance

        Returns:
            Generated instance if successful, None otherwise
        """
        is_completed = False
        for _ in range(self.config.max_retries):
            try:

                # 1. Prepare prompt
                if isinstance(self.prompt_loader, InstanceGenerationPromptLoader):
                    prompt_result = self.prompt_loader.prepare()
                else:
                    prompt_result = self.prompt_loader.prepare(index)

                # 2. Get completion
                messages = [
                    {"role": "system", "content": self.config.system_message},
                    {"role": "user", "content": prompt_result.prompt},
                ]

                completion = self.llm.chat(messages, **self.sampling_params)
                # 3. Check if completion is successful
                is_completed = self.validate_finish_reason(completion)

                # 4. Parse model output & Validate result
                if is_completed:
                    try:
                        teacher_model_output = completion["choices"][0]["message"]["content"]
                    except:
                        try:
                            teacher_model_output = completion.choices[0].message.content
                        except:
                            raise ValueError("Invalid completion format")
                    parsed_result = self.parser.parse(
                        prompt_result.prompt, teacher_model_output, self.placeholder_formats
                    )
                    instruction = parsed_result["instruction"]
                    response = parsed_result["response"]

                    if parsed_result is not None:
                        is_valid = self.validator.validate(instruction, response)

                        if is_valid is True:
                            # Add metadata
                            try:
                                parsed_result["metadata"] = {
                                    "model": completion.get("model", "unknown"),
                                    **prompt_result.metadata,
                                }
                            except:
                                try:
                                    parsed_result["metadata"] = {"model": completion.model, **prompt_result.metadata}
                                except:
                                    raise ValueError("Invalid metadata format")

                            # Add index for non-InstanceGenerationPromptLoader
                            if not isinstance(self.prompt_loader, InstanceGenerationPromptLoader):
                                parsed_result["index"] = index

                            # 5. For InstanceGenerationPromptLoader, update seed_data with thread safety
                            if isinstance(self.prompt_loader, InstanceGenerationPromptLoader):
                                if hasattr(self.prompt_loader, "seed_data"):
                                    with self._seed_data_lock:
                                        self.prompt_loader.seed_data.append(parsed_result)

                            return parsed_result
            except Exception as e:
                if _ < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    print("[ERROR]", e)
        if is_completed is True:
            print("Failed example:")
            print(completion)
        return None

    def run(
        self,
        num_instances: int,
        output_file: Union[str, Path],
        cache_file: Optional[Union[str, Path]] = None,
        num_threads: Optional[int] = None,
    ) -> List[Dict]:
        """Generate multiple instances with progress tracking, retry logic, and caching

        Args:
            num_instances: Number of instances to generate
            cache_file: Optional path to cache file for intermediate saving

        Returns:
            List of generated instances
        """

        # Check if vLLM is being used
        if self.llm.__class__.__name__ == "LocalVLLM":
            return self.run_vllm(num_instances, output_file, cache_file)

        # Initialize or load existing results
        results = []

        # Add a lock for thread-safe results list modification
        results_lock = Lock()

        if cache_file is not None:
            cache_path = Path(cache_file).expanduser()
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            if cache_path.exists():
                if cache_path.suffix == ".jsonl":
                    try:
                        results = []
                        with cache_path.open("r") as f:
                            for line in f:
                                results.append(json.loads(line))

                        print(f"Loaded {len(results)} existing results from cache")

                        # Trim results if more than num_instances
                        if len(results) >= num_instances:
                            results = results[:num_instances]
                            # Save to output file and delete cache
                            results = [
                                {
                                    "instruction": item["instruction"],
                                    "response": item["response"],
                                    "config": self.llm.model_name,
                                }
                                for item in results
                            ]
                            with open(output_file, "w") as f:
                                json.dump(results, f, indent=4)
                            cache_path.unlink()
                            return results

                    except json.JSONDecodeError:
                        print("Warning: Cache file exists but is not valid JSONL. Starting fresh.")
                else:
                    print("Warning: Cache file must be a .jsonl file. Starting fresh.")
        else:
            cache_path = Path(output_file.replace(".json", ".jsonl")).expanduser()

        # Calculate remaining instances to generate
        instances_to_generate = num_instances - len(results)

        if instances_to_generate <= 0:
            print(f"Already generated {len(results)} instances, target is {num_instances}")
            return results

        if isinstance(self.prompt_loader, InstanceGenerationPromptLoader):
            if hasattr(self.prompt_loader, "seed_data"):
                self.prompt_loader.seed_data.extend(results)
        else:
            # Add index to seed data if not present
            for i, item in enumerate(self.prompt_loader.seed_data):
                if "index" not in item:
                    item["index"] = i

            # Get indices of unprocessed items
            processed_indices = {result["index"] for result in results if "index" in result}
            index_list_to_process = [i for i in range(num_instances) if i not in processed_indices]

        if num_threads is not None and num_threads > 1:
            # Parallel generation using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(num_threads, instances_to_generate)) as executor:
                futures = []

                with tqdm(total=instances_to_generate, disable=not self.config.show_progress) as pbar:
                    # Submit all tasks at once
                    if isinstance(self.prompt_loader, InstanceGenerationPromptLoader):
                        futures = [executor.submit(self.run_single) for _ in range(instances_to_generate)]
                    else:
                        futures = [executor.submit(self.run_single, index) for index in index_list_to_process]

                    # Process completed futures
                    for future in as_completed(futures):
                        result = future.result()
                        if result is not None:
                            with results_lock:
                                results.append(result)
                                pbar.update(1)

                                # Save intermediate results
                                try:
                                    with cache_path.open("a") as f:
                                        save_result = {
                                            "instruction": result["instruction"],
                                            "response": result["response"],
                                            "config": self.llm.model_name,
                                        }
                                        if "index" in result:
                                            save_result["index"] = result["index"]
                                        json.dump(save_result, f)
                                        f.write("\n")
                                except Exception as e:
                                    print(f"\nWarning: Failed to save to cache file: {str(e)}")

        else:
            # Sequential generation
            with tqdm(total=instances_to_generate, disable=not self.config.show_progress) as pbar:
                if isinstance(self.prompt_loader, InstanceGenerationPromptLoader):
                    for _ in range(instances_to_generate):
                        result = self.run_single()
                        if result is not None:
                            results.append(result)
                            pbar.update(1)

                            # Save intermediate results
                            try:
                                with cache_path.open("a") as f:
                                    save_result = {
                                        "instruction": result["instruction"],
                                        "response": result["response"],
                                        "config": self.llm.model_name,
                                    }
                                    if "index" in result:
                                        save_result["index"] = result["index"]
                                    json.dump(save_result, f)
                                    f.write("\n")
                            except Exception as e:
                                print(f"\nWarning: Failed to save to cache file: {str(e)}")
                else:
                    for index in index_list_to_process:
                        result = self.run_single(index)
                        if result is not None:
                            results.append(result)
                            pbar.update(1)

                            # Save intermediate results
                            try:
                                with cache_path.open("a") as f:
                                    save_result = {
                                        "instruction": result["instruction"],
                                        "response": result["response"],
                                        "config": self.llm.model_name,
                                        "index": result["index"],
                                    }
                                    json.dump(save_result, f)
                                    f.write("\n")
                            except Exception as e:
                                print(f"\nWarning: Failed to save to cache file: {str(e)}")

        # Remove index from final results
        results = [
            {"instruction": item["instruction"], "response": item["response"], "config": self.llm.model_name}
            for item in results
        ]

        unfinished_instances = num_instances - len(results)
        if unfinished_instances > 0:
            print(
                f"\nWarning: {unfinished_instances} instances failed to complete after {self.config.max_retries} retries"
            )
            print(f"Cache file preserved at: {cache_path}")
            return results

        if cache_path.exists():
            cache_path.unlink()

        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

        return results

    def run_vllm(
        self,
        num_instances: int,
        output_file: Union[str, Path],
        cache_file: Optional[Union[str, Path]] = None,
        batch_size: int = 10,
    ) -> List[Dict]:
        """Generate multiple instances using vLLM's batch inference capabilities

        Args:
            num_instances: Number of instances to generate
            output_file: Path to save final results
            cache_file: Optional path to cache file for intermediate saving
            batch_size: Number of instances to generate in each batch

        Returns:
            List of generated instances
        """
        # Initialize or load existing results
        results = []

        if cache_file is not None:
            cache_path = Path(cache_file).expanduser()
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            if cache_path.exists():
                if cache_path.suffix == ".jsonl":
                    try:
                        results = []
                        with cache_path.open("r") as f:
                            for line in f:
                                results.append(json.loads(line))

                        print(f"Loaded {len(results)} existing results from cache")

                        # Trim results if more than num_instances
                        if len(results) >= num_instances:
                            results = results[:num_instances]
                            # Save to output file and delete cache
                            results = [
                                {
                                    "instruction": item["instruction"],
                                    "response": item["response"],
                                    "config": self.llm.model_name,
                                }
                                for item in results
                            ]
                            with open(output_file, "w") as f:
                                json.dump(results, f, indent=4)
                            cache_path.unlink()
                            return results

                    except json.JSONDecodeError:
                        print("Warning: Cache file exists but is not valid JSONL. Starting fresh.")
                else:
                    print("Warning: Cache file must be a .jsonl file. Starting fresh.")
        else:
            cache_path = Path(output_file.replace(".json", ".jsonl")).expanduser()

        # Calculate remaining instances to generate
        instances_to_generate = num_instances - len(results)

        if instances_to_generate <= 0:
            print(f"Already generated {len(results)} instances, target is {num_instances}")
            return results

        # Prepare indices for generation
        if isinstance(self.prompt_loader, InstanceGenerationPromptLoader):
            total_batches = (instances_to_generate + batch_size - 1) // batch_size
            all_indices = list(range(instances_to_generate))
        else:
            processed_indices = {result["index"] for result in results if "index" in result}
            all_indices = [i for i in range(num_instances) if i not in processed_indices]
            total_batches = (len(all_indices) + batch_size - 1) // batch_size

        # Process all batches
        with tqdm(total=instances_to_generate, disable=not self.config.show_progress) as pbar:
            for batch_start in range(0, len(all_indices), batch_size):
                batch_indices = all_indices[batch_start : batch_start + batch_size]

                # Prepare prompts for the batch
                messages_list = []
                prompt_results = []

                for idx in batch_indices:
                    if isinstance(self.prompt_loader, InstanceGenerationPromptLoader):
                        prompt_result = self.prompt_loader.prepare()
                    else:
                        prompt_result = self.prompt_loader.prepare(idx)

                    messages = [
                        {"role": "system", "content": self.config.system_message},
                        {"role": "user", "content": prompt_result.prompt},
                    ]
                    messages_list.append(messages)
                    prompt_results.append(prompt_result)

                # Get batch completions using vLLM
                batch_outputs = self.llm.chat(
                    messages=messages_list, sampling_params=self.sampling_params, use_tqdm=False
                )

                # Process outputs
                for i, (output, prompt_result) in enumerate(zip(batch_outputs, prompt_results)):
                    try:
                        teacher_model_output = output.outputs[0].text
                        prompt = output.prompt

                        # Parse and validate
                        parsed_result = self.parser.parse(prompt, teacher_model_output, self.placeholder_formats)
                        instruction = parsed_result["instruction"]
                        response = parsed_result["response"]

                        if parsed_result is not None:
                            is_valid = self.validator.validate(instruction, response)

                            if is_valid:
                                # Add metadata
                                parsed_result["metadata"] = {"model": self.llm.model_name, **prompt_result.metadata}

                                # Add index for non-InstanceGenerationPromptLoader
                                if not isinstance(self.prompt_loader, InstanceGenerationPromptLoader):
                                    parsed_result["index"] = batch_indices[i]

                                results.append(parsed_result)
                                pbar.update(1)

                                # Save intermediate results
                                try:
                                    with cache_path.open("a") as f:
                                        save_result = {
                                            "instruction": parsed_result["instruction"],
                                            "response": parsed_result["response"],
                                            "config": self.llm.model_name,
                                        }
                                        if "index" in parsed_result:
                                            save_result["index"] = parsed_result["index"]
                                        json.dump(save_result, f)
                                        f.write("\n")
                                except Exception as e:
                                    print(f"\nWarning: Failed to save to cache file: {str(e)}")

                    except Exception as e:
                        print(f"Error processing batch item {i}: {str(e)}")
                        continue

        # Remove index from final results
        results = [
            {"instruction": item["instruction"], "response": item["response"], "config": self.llm.model_name}
            for item in results
        ]

        unfinished_instances = num_instances - len(results)
        if unfinished_instances > 0:
            print(f"\nWarning: {unfinished_instances} instances failed to complete")
            print(f"Cache file preserved at: {cache_path}")
            return results

        if cache_path.exists():
            cache_path.unlink()

        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

        return results
