import json
import os
from abc import ABC
from typing import Dict, List, Optional, Tuple

import numpy as np
from prometheus_eval import SCORE_RUBRIC_TEMPLATE, PrometheusEval
from transformers import AutoModelForCausalLM, AutoTokenizer

rubrics = {
    "general": {
        "complexity": {
            "criteria": "How complex and challenging is the given instruction to answer perfectly?",
            "score1_description": "The instruction requires only factual knowledge, without any need for reasoning or critical thinking. A straightforward, single-step response suffices.",
            "score2_description": "The instruction requires some reasoning, such as explaining a concept involving multiple simple ideas, solving a straightforward problem, or providing a response that involves a few logical steps, though still simple in nature.",
            "score3_description": "The instruction requires a substantial amount of reasoning and the integration of multiple related concepts. Answering it accurately involves a multi-step process and may require intermediate-level knowledge or analytical thinking.",
            "score4_description": "The instruction requires advanced reasoning, demanding deep understanding of complex concepts or substantial problem-solving. Answering it requires carefully navigating multiple interrelated ideas or steps, often involving specialized knowledge or sophisticated analytical skills.",
            "score5_description": "The instruction is exceptionally challenging and requires high-level reasoning or novel problem-solving. It involves extensive conceptual understanding, abstraction, and potentially innovative thinking, with substantial effort required to arrive at an accurate and complete answer.",
        },
        "quality": {
            "criteria": "Does the response consider a wide range of factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail?",
            "score1_description": "The response is not helpful at all or seems helpful on the surface but is actually incorrect such as including incorrect information, naive miscalculations, or unexecutable code.",
            "score2_description": "The response contains some relevant or helpful information, but also has major flaws interms of factuality, accuracy, and relevance.",
            "score3_description": "The response is mostly correct but minor flaws regarding factuality, accuracy, and relevance still exists, while it is overall an okay response.",
            "score4_description": "The response is accurate, relevant, and helpful, although there are some slight improvements that could be made when an expert evaluates the response.",
            "score5_description": "The response is excellent. It is completely factual, accurate, relevant, and helpful, demonstrating a high degree of depth and creativity.",
        },
    },
    "math": {
        "complexity": {
            "criteria": "How complex and challenging is the math problem to solve?",
            "score1_description": "The problem requires only simple operations or direct application of a single, basic concept. Minimal reasoning is needed, and the solution follows immediately from applying a known rule or formula.",
            "score2_description": "The problem requires basic reasoning and involves applying a familiar formula or concept with slight variation. It may involve a straightforward multi-step process, but each step is clear and relies on commonly used methods.",
            "score3_description": "The problem requires moderate reasoning, combining multiple concepts that interact in a meaningful way. Solving it involves several steps and may require logical sequencing or some abstraction, but the approach is approachable with a solid foundational understanding.",
            "score4_description": " The problem demands advanced reasoning, involving multiple interdependent concepts that require careful coordination. Solution steps are less obvious, requiring critical thinking and possibly choosing between multiple solution paths. Solving the problem involves more abstract reasoning or creative application of concepts.",
            "score5_description": "The problem is extremely complex and demands sophisticated reasoning and problem-solving skills. It may involve novel combinations of concepts, intricate logical chains, or innovative approaches to solve. This level typically requires significant abstraction, exploration of unconventional methods, and flexibility in adapting mathematical tools.",
        },
        "quality": {
            "criteria": "Does the solution demonstrate mathematical correctness, reasoning, clarity, and precision?",
            "score1_description": "The solution is incorrect or mathematically flawed, with major errors in reasoning, calculations, or logic, making the answer unusable.",
            "score2_description": "The solution contains relevant or partially correct information, but has significant errors in calculations or reasoning that substantially affect the result.",
            "score3_description": "The solution is mostly correct but may contain minor mistakes or gaps in reasoning. The overall structure and approach are sound, but some calculations or logic may need refinement.",
            "score4_description": "The solution is correct, well-reasoned, and clear, though there may be slight room for improvement or minor refinements to become a perfect solution to the problem.",
            "score5_description": "The solution is excellent, fully correct, and demonstrates a high level of mathematical precision, clarity, and creativity, with well-articulated reasoning and no errors.",
        },
    },
    "code": {
        "complexity": {
            "criteria": "How complex and challenging is the coding problem to solve?",
            "score1_description": "The problem involves implementing simple functionality or a direct operation. It requires minimal logic, with a straightforward approach and no complex decision-making.",
            "score2_description": "The problem requires basic control flow, such as using loops or conditional statements. The logic is clear and sequential, with minimal interaction between different parts of the code.",
            "score3_description": "The problem involves intermediate logic, combining multiple programming constructs and requiring a coherent structure. Solving it requires handling a sequence of steps with basic data manipulation, but follows a familiar, manageable approach.",
            "score4_description": "The problem demands advanced reasoning and use of complex data structures or algorithms. It involves non-trivial interactions, such as managing multiple components and optimizing for efficiency. The solution requires significant algorithmic thinking and structured problem decomposition.",
            "score5_description": "The problem is extremely complex, requiring sophisticated algorithm design, efficient data handling, and advanced techniques. It demands innovative approaches, with intricate component interactions and constraints that need careful optimization. Solving it typically requires deep problem-solving skills and adaptability across programming paradigms.",
        },
        "quality": {
            "criteria": "How effective, efficient, and logically sound is the code solution, focusing on performance, executability, and correctness?",
            "score1_description": "The code contains fundamental logic or syntax errors, making it incorrect or unexecutable. It fails to complete the intended task or produces entirely incorrect outputs.",
            "score2_description": "The code is partially functional but contains major logic errors or inefficiencies that significantly impact performance or correctness. It may run but produces incorrect or incomplete results.",
            "score3_description": "The code is mostly correct and executable, though there may be minor logic issues, inefficiencies, or suboptimal use of data structures or algorithms. The solution functions as intended, but improvements could make it more efficient or robust.",
            "score4_description": "The code is fully correct, functional, and reasonably efficient. It completes the task as intended, balancing performance with logical soundness. Minor optimizations could still enhance performance.",
            "score5_description": "The code is fully correct, optimally efficient, and logically robust, providing the best possible performance for the task. It executes flawlessly without errors or any significant room for improvement.",
        },
    },
}


class IntrinsicEvaluator(ABC):
    """Abstract base class for intrinsic evaluators"""

    @abstractmethod
    def evaluate(self, instances: List[Dict[str, str]]) -> float:
        """Evaluate the instances"""
        pass


class ResponseQualityEvaluator(IntrinsicEvaluator):
    """Evaluator for response quality"""

    def __init__(self, model_path: str):
        self.judge = PrometheusEval(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def llm_as_a_judge_evaluate(self, domain: str, instances: List[Dict[str, str]]) -> float:
        instructions = []
        responses = []
        for instance in instances:
            instructions.append(instance["instruction"])
            responses.append(instance["response"])

        rubric = SCORE_RUBRIC_TEMPLATE.format(**rubrics[domain]["quality"])

        quality_feedback, quality_score = self.judge.absolute_grade(
            instructions=instructions, responses=responses, rubric=rubric
        )

        return quality_feedback, quality_score, np.mean(quality_score)


class InstructionComplexityEvaluator(IntrinsicEvaluator):
    """Evaluator for instruction complexity"""

    def __init__(self, model_path: str):

        self.judge = PrometheusEval(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()

    def llm_as_a_judge_evaluate(self, domain: str, instances: List[Dict[str, str]]) -> float:
        instructions = []
        for instance in instances:
            instructions.append(instance["instruction"])

        complexity_rubric = SCORE_RUBRIC_TEMPLATE.format(**rubrics[domain]["complexity"])

        complexity_feedback, complexity_score = self.judge.absolute_grade(
            instructions=[f"Your task is to generate a {domain} problem that is not too generic or easy to solve."]
            * len(instructions),
            responses=instructions,
            rubric=complexity_rubric,
        )

        return complexity_feedback, complexity_score, np.mean(complexity_score)

    def perplexity_evaluate(self, instances: List[Dict[str, str]], batch_size: int = 8) -> float:
        total_perplexity = 0.0
        results = []
        temp_file = "temp_results.jsonl"

        with torch.no_grad():
            # Process instances in batches
            for i in range(0, len(instances), batch_size):
                batch = instances[i : i + batch_size]

                # Prepare batch inputs
                batch_inputs = []
                batch_instruction_lens = []
                for instance in batch:
                    instruction = instance["instruction"]
                    response = instance["response"]
                    batch_inputs.append(instruction + response)
                    batch_instruction_lens.append(len(self.tokenizer(instruction)["input_ids"]))

                # Tokenize batch
                inputs = self.tokenizer(batch_inputs, return_tensors="pt", padding=True).to(self.device)
                labels = inputs["input_ids"].clone()

                # Mask out loss for instruction tokens for each sequence in batch
                for j, inst_len in enumerate(batch_instruction_lens):
                    labels[j, :inst_len] = -100

                # Calculate loss
                outputs = self.model(**inputs, labels=labels)
                losses = outputs.loss.item()

                # Process each item in batch
                for j, instance in enumerate(batch):
                    perplexity = torch.exp(torch.tensor(losses)).item()
                    total_perplexity += perplexity

                    result = {
                        "instruction": instance["instruction"],
                        "response": instance["response"],
                        "perplexity": perplexity,
                    }

                    with open(temp_file, "a") as f:
                        f.write(json.dumps(result) + "\n")

                    results.append(result)

        # Save final results and delete temp file
        with open("evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)

        if os.path.exists(temp_file):
            os.remove(temp_file)

        print(f"Total perplexity: {total_perplexity}")
        return total_perplexity / len(instances)
