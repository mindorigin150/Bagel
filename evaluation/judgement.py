# evaluation/judgement.py
import abc
import os
import requests
import json
import concurrent.futures
import time
from tqdm import tqdm
from typing import Dict, TypedDict, Any, List, Tuple
from .inference_engine import BaseInferenceEngine

# --- 1. Core Data Structures and Exceptions ---

class JudgementResult(TypedDict):
    """Standardized result format for all judges."""
    is_correct: bool
    reason: str

class JudgementError(Exception):
    """Custom exception for critical, unrecoverable judgement failures."""
    pass

# --- 2. Shared Prompt Template ---

# Centralized prompt template to ensure consistency across all judges.
# Using .format() with named placeholders {gt_answer} and {model_answer}.
"""This prompt is still not enough"""
# JUDGE_PROMPT_TEMPLATE = """Please act as an impartial and intelligent judge. Your task is to evaluate the quality of an AI assistant's response based on a provided reference answer.
# Your evaluation must follow these crucial rules:
# 1.  **Focus on Semantic Correctness, Not Literal Matching**: Your primary goal is to determine if the assistant's answer correctly identifies the core information from the reference answer. Do not penalize for minor formatting differences, verbosity, or extra explanations, as long as the core answer is correct.
#     *   **Example**: If the reference answer is "A", the following assistant answers are all considered **CORRECT**:
#         *   "A. Yes"
#         *   "A"
#         *   "The correct answer is A."
#         *   "<think>...</think><answer>A</answer>"
#         *   "Based on my analysis, the answer is A. Yes."
# 2.  **Evaluate the Reasoning Process**: If the final answer part is missing, incorrect, or ambiguous, you MUST inspect the assistant's entire response (including any text within `<think>` tags or similar). If the reasoning process clearly and correctly identifies the right answer before the final output, it should be considered **CORRECT**.
#     *   **Example**: The reference answer is "A". The assistant says: "<think>After analyzing the images, it's clear option A is the correct one. But let me double-check the details of image 2 one more time...</think>" (output truncated here). This should be judged as **CORRECT** because the reasoning identified the right answer.
# 3.  **Strict Output Format**: You must return your verdict as a single JSON object. The JSON object should have two keys:
#     *   `"is_correct"`: A boolean value (`true` or `false`).
#     *   `"reason"`: A brief, clear justification for your verdict, especially if you had to rely on Rule #2.
# [BEGIN DATA]
# ***
# [Reference Answer]:
# {gt_answer}
# ***
# [Assistant's Answer]:
# {model_answer}
# ***
# [END DATA]
# Your verdict:
# """
"""An enhanced version"""
JUDGE_PROMPT_TEMPLATE = """Please act as an impartial and intelligent judge. Your task is to evaluate the quality of an AI assistant's response based on a provided reference answer.

Your evaluation must follow these crucial rules:

1.  **Focus on Semantic Correctness, Not Literal Matching**: Your primary goal is to determine if the assistant's answer correctly identifies the core information from the reference answer. Do not penalize for minor formatting differences, verbosity, or extra explanations, as long as the core answer is correct.
    *   **Example 1**: If the reference answer is "A", the following assistant answers are all considered **CORRECT**:
        *   "A. Yes"
        *   "A"
        *   "The correct answer is A."
        *   "<think>...</think><answer>A</answer>"
    *   **Example 2 (Crucial Rule)**: If the reference answer is "A", an assistant answer like "<answer>A. Diagonally forward and left</answer>" is also **CORRECT**. The core answer 'A' is present and correct. The additional text is just a valid explanation and should not be penalized.

2.  **Evaluate the Reasoning Process**: If the final answer part is missing, incorrect, or ambiguous, you MUST inspect the assistant's entire response (including any text within `<think>` tags). If the reasoning process clearly and correctly identifies the right answer before the final output, it should be considered **CORRECT**.
    *   **Example**: The reference answer is "A". The assistant says: "<think>After analyzing the images, it's clear option A is the correct one. But let me double-check...</think>" (output is truncated). This should be judged as **CORRECT**.

3.  **Mandatory Thinking Process and Strict Output Format**: You MUST first write down your step-by-step reasoning process inside `<reasoning>` tags. After the reasoning, you MUST return your verdict as a single JSON object.

    Your reasoning MUST follow these steps:
    1.  Identify the core information in the [Reference Answer].
    2.  Identify the core information in the [Assistant's Answer]. Check both the final output and any thinking process.
    3.  Compare the core information from both answers. Based on Rule #1 and #2, determine if they are semantically identical. Conclude with a clear "Therefore, the assistant's answer is correct/incorrect."

    The final JSON object must have two keys:
    *   `"is_correct"`: A boolean value (`true` or `false`).
    *   `"reason"`: A brief, clear justification for your verdict, which must be consistent with your reasoning process.

[BEGIN DATA]
***
[Reference Answer]:
{gt_answer}
***
[Assistant's Answer]:
{model_answer}
***
[END DATA]

Your verdict:
"""


# --- 3. Abstract Base Class ---

class BaseJudge(abc.ABC):
    """
    Abstract interface defining the contract for all judge implementations.
    """
    @abc.abstractmethod
    def judge(self, model_answer: str, ground_truth_answer: str) -> JudgementResult:
        """
        Compares a model's answer to the ground truth and returns a structured judgment.
        """
        raise NotImplementedError

    def batch_judge(self, items: List[Tuple[str, str]]) -> List[JudgementResult]:
        """
        Evaluates a batch of items. Default implementation calls judge() in a loop.
        Subclasses can override this for more efficient batching.
        """
        # A simple, default implementation. It works, but it's not efficient.
        return [self.judge(model_answer, gt_answer) for model_answer, gt_answer in items]


def _parse_llm_json_output(model_output: str) -> JudgementResult:
    """
    A robust utility function to parse JSON from a model's string output.
    This logic is shared by all judges that get a JSON response from an LLM.
    """
    try:
        # Find the first '{' and the last '}' to make parsing more robust
        # against models that output extra text before or after the JSON.
        start_index = model_output.find('{')
        end_index = model_output.rfind('}')
        
        if start_index == -1 or end_index == -1:
            raise ValueError("No JSON object found in model output.")
        
        json_str = model_output[start_index : end_index + 1]
        parsed_json = json.loads(json_str)
        
        is_correct = parsed_json.get('is_correct')
        if not isinstance(is_correct, bool):
            raise ValueError("'is_correct' key is missing or not a boolean.")
            
        return {
            "is_correct": is_correct,
            "reason": str(parsed_json.get("reason", "No reason provided."))
        }
        
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        # If parsing fails, don't crash. Return a structured error.
        # This is the essence of robust code.
        return {
            "is_correct": False,
            "reason": f"Failed to parse model judgement. Error: {e}. Raw output: '{model_output[:100]}...'"
        }

# --- 4. Concrete Judge Implementations ---
class APIModelJudge(BaseJudge):
    """Judges correctness by calling an external API (e.g., OpenAI, Anthropic)."""
    def __init__(self, endpoint: str, api_key: str = None, timeout: int = 30, model: str = "gpt-4o", max_workers: int = 4, requests_per_minute: int = 29):
        if not endpoint:
            raise ValueError("API endpoint cannot be empty.")
        self.endpoint = endpoint
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.timeout = timeout
        self.model = model
        self.max_workers = max_workers

        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive.")
        self.request_interval = 60.0 / requests_per_minute

    def judge(self, model_answer: str, ground_truth_answer: str) -> JudgementResult:
        prompt = JUDGE_PROMPT_TEMPLATE.format(gt_answer=ground_truth_answer, model_answer=model_answer)
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        try:
            response = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content']
            return _parse_llm_json_output(content)
            
        except requests.exceptions.RequestException as e:
            return {"is_correct": False, "reason": f"API request failed: {e}"}
        except (KeyError, IndexError) as e:
            return {"is_correct": False, "reason": f"Invalid API response format: {e}"}

    def batch_judge(self, items: List[Tuple[str, str]]) -> List[JudgementResult]:
        """
        Efficiently judges a batch of items with rate limiting and progress feedback.
        It has two progress bars: one for submitting tasks, one for collecting results.
        """
        if not items:
            return []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # --- 阶段 1: 提交任务，带进度条 ---
            futures = []
            # 用 tqdm 包装 'items' 迭代器
            for model_answer, gt_answer in tqdm(items, desc="Submitting API tasks", unit="task"):
                future = executor.submit(self.judge, model_answer, gt_answer)
                futures.append(future)
                time.sleep(self.request_interval)
            # --- 阶段 2: 收集结果，也带进度条 ---
            results = []
            # 用 tqdm 包装 'futures' 迭代器
            for future in tqdm(concurrent.futures.as_completed(futures), desc="Collecting API results", total=len(futures), unit="result"):
                try:
                    results.append(future.result())
                except Exception as exc:
                    error_result = {
                        "is_correct": False,
                        "reason": f"An unexpected error occurred during future execution: {exc}"
                    }
                    results.append(error_result)
        
        # as_completed's results are not ordered, so we need to reorder them
        # Let's fix this mess from the previous version. The order of items must be preserved.
        # The code above using as_completed is WRONG because it breaks the order.
        # This is the correct way, which I provided before. Let's stick to it.
        # My apologies, as_completed is a distraction here. Let's get back to the simple, correct code.
        # === CORRECTED AND FINAL IMPLEMENTATION ===
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Stage 1: Submission with progress
            futures = []
            for model_answer, gt_answer in tqdm(items, desc="Submitting API tasks", unit="task"):
                future = executor.submit(self.judge, model_answer, gt_answer)
                futures.append(future)
                time.sleep(self.request_interval)
            
            # Stage 2: Collection with progress. Iterating the futures list directly PRESERVES ORDER.
            results = []
            for future in tqdm(futures, desc="Collecting API results", unit="result"):
                try:
                    results.append(future.result())
                except Exception as exc:
                    error_result = {
                        "is_correct": False,
                        "reason": f"An unexpected error occurred during future execution: {exc}"
                    }
                    results.append(error_result)
        
        return results


class LocalJudge(BaseJudge):
    """Judges correctness using a local model via a unified inference engine."""
    
    _template = "[INST] {system_prompt}\n{prompt} [/INST]"
    _system_prompt = "You are a helpful and precise assistant for checking the quality of AI responses."

    def __init__(self, engine: BaseInferenceEngine):
        """
        Args:
            engine: An initialized instance of a class derived from BaseInferenceEngine.
        """
        self.engine = engine

    def _format_prompt(self, model_answer: str, ground_truth_answer: str) -> str:
        """Formats the final prompt including the system message and instruction wrapper."""
        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            gt_answer=ground_truth_answer,
            model_answer=model_answer
        )
        return self._template.format(system_prompt=self._system_prompt, prompt=judge_prompt)

    def judge(self, model_answer: str, ground_truth_answer: str) -> JudgementResult:
        """Generates a single judgement using the local inference engine."""
        prompt = self._format_prompt(model_answer, ground_truth_answer)
        responses = self.engine.batch_generate([prompt])
        return _parse_llm_json_output(responses[0])

    def batch_judge(self, items: List[Tuple[str, str]]) -> List[JudgementResult]:
        """
        Efficiently judges a batch of items by sending them all to the engine at once.
        This overrides the inefficient default implementation in BaseJudge.
        """
        # This is the real advantage of this class.
        prompts = [self._format_prompt(ans, gt) for ans, gt in items]
        responses = self.engine.batch_generate(prompts)
        return [_parse_llm_json_output(resp) for resp in responses]

