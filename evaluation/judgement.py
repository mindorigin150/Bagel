# evaluation/judgement.py
import abc
import os
import requests
import json
import concurrent.futures
import time
from tqdm import tqdm
from typing import Dict, TypedDict, Any, List, Tuple, Generator, final
from .inference_engine import BaseInferenceEngine
from transformers import AutoTokenizer

# --- 1. Core Data Structures and Exceptions ---

class JudgementResult(TypedDict):
    """Standardized result format for all judges."""
    is_correct: bool
    reason: str

class JudgementError(Exception):
    """Custom exception for critical, unrecoverable judgement failures."""
    pass

# --- 2. Shared Prompt Template ---

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
[Question]:
{question}
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
    def judge(self, question: str, model_answer: str, ground_truth_answer: str) -> JudgementResult:
        """
        Compares a model's answer to the ground truth and returns a structured judgment.
        """
        raise NotImplementedError

    def batch_judge(self, items: List[Tuple[str, str]]) -> Generator[Tuple[int, JudgementResult], Any, None]:
        """
        Default generator implementation. Evaluates items one by one and yields results.
        This is correct, but inefficient. Subclasses SHOULD override this.
        """
        for i, (question, model_answer, gt_answer) in enumerate(items):
            yield i, self.judge(question, model_answer, gt_answer)


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
    """
    Judges correctness by calling an external API (e.g., OpenAI, Anthropic).
    This version operates sequentially with rate limiting and retry logic.
    """
    def __init__(self, endpoint: str, api_key: str = None, timeout: int = 120, model: str = "gpt-4o", max_retries: int = 10, requests_per_minute: int = 60):
        """
        Args:
            endpoint: The API endpoint URL.
            api_key: The API key for authentication.
            timeout: Request timeout in seconds.
            model: The model name to use.
            max_retries: The maximum number of times to retry a failed API request.
            requests_per_minute: The number of requests to allow per minute for rate limiting.
        """
        if not endpoint:
            raise ValueError("API endpoint cannot be empty.")
        self.endpoint = endpoint
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.timeout = timeout
        self.model = model
        self.max_retries = max_retries
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive.")
        # calculate interval for each request
        self.request_interval = 60.0 / requests_per_minute

    def judge(self, question: str, model_answer: str, ground_truth_answer: str) -> JudgementResult:
        """
        Sends a single, robust request to the API, with built-in retries for network errors.
        """
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            gt_answer=ground_truth_answer,
            model_answer=model_answer
        )
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        payload = {
            "model": self.model,
            "messages": messages,
        }
        
        last_exception = None
        
        # Retry logic for each request
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=self.timeout)
                response.raise_for_status()  # 如果状态码是 4xx 或 5xx，则抛出异常
                content = response.json()['choices'][0]['message']['content']
                return _parse_llm_json_output(content)
            
            except requests.exceptions.RequestException as e:
                # 捕获网络层面的错误 (e.g., timeout, connection error)
                last_exception = e
                print(f"⚠️ API request failed (attempt {attempt + 1}/{self.max_retries + 1}). Error: {e}")
                if attempt < self.max_retries:
                    time.sleep(1)  # a short break before retrying
                continue
            
            except (KeyError, IndexError) as e:
                # catch invalid API response format, which is normally permanent and no need to retry
                return {"is_correct": False, "reason": f"Invalid API response format: {e}"}
        
        # If all attempts failed
        return {"is_correct": False, "reason": f"API request failed after {self.max_retries + 1} attempts. Last error: {last_exception}"}

    def batch_judge(self, items: List[Tuple[str, str, str]]) -> Generator[Tuple[int, JudgementResult], Any, None]:
        """
        Judges a batch of items sequentially with rate limiting.
        This implementation uses a simple for loop, without concurrency.
        """
        if not items:
            return

        for i, (question, model_answer, gt_answer) in enumerate(tqdm(items, desc="Judging items sequentially", unit="item")):
            result = self.judge(question, model_answer, gt_answer)
            yield i, result

            if i < len(items) - 1:
                time.sleep(self.request_interval)


class LocalJudge(BaseJudge):
    """Judges correctness using a local model via a unified inference engine."""
    def __init__(self, engine: BaseInferenceEngine, max_retries: int = 10):
        """
        Args:
            engine: An initialized instance of a class derived from BaseInferenceEngine.
            max_retries: When output of model cannot be parsed, the max retry times.
        """
        self.engine = engine
        self.model = engine.model
        self.tokenizer = AutoTokenizer.from_pretrained(engine.model_path, trust_remote_code=True)
        self.max_retries = max_retries

    def _format_prompt(self, question: str, model_answer: str, ground_truth_answer: str) -> str:
        """Formats the final prompt including the system message and instruction wrapper."""
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            gt_answer=ground_truth_answer,
            model_answer=model_answer
        )
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        formatted_prompt_string = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        return formatted_prompt_string

    def judge(self, question: str, model_answer: str, ground_truth_answer: str) -> JudgementResult:
        """Generates a single judgement using the local inference engine."""
        prompt = self._format_prompt(question, model_answer, ground_truth_answer)
        last_result = None
        for attempt in range(self.max_retries + 1):
            responses = self.engine.batch_generate([prompt])
            result = _parse_llm_json_output(responses[0])
            
            if "Failed to parse" not in result["reason"]:
                return result
            
            last_result = result
            if attempt < self.max_retries:
                print(f"⚠️ Retrying judgement due to parsing failure (attempt {attempt + 1}/{self.max_retries})...")
                
        # return the last result if all attempts failed
        print(f"❌ Judgement failed after {self.max_retries} retries.")
        return last_result

    def batch_judge(self, items: List[Tuple[str, str]]) -> Generator[Tuple[int, JudgementResult], Any, None]:
        """
        Efficiently judges a batch of items by sending them all to the engine at once,
        then yields the results to maintain a consistent generator interface.
        """
        if not items:
            return
        
        # 1. put all original items and their indexes into todo list
        items_to_judge = [(i, question, ans, gt) for i, (question, ans, gt) in enumerate(items)]
        
        for attempt in range(self.max_retries + 1):
            if not items_to_judge:
                break
            print(f"🚀 Starting judgement batch... Attempt {attempt + 1}, items remaining: {len(items_to_judge)}")
            
            # prepare prompts for current batch
            prompts = [self._format_prompt(question, ans, gt) for _, question, ans, gt in items_to_judge]
            responses = self.engine.batch_generate(prompts)
            
            # prepare an empty list for next retry
            failed_items_for_next_round = []
            # process results for current batch
            for i, response in enumerate(responses):
                original_index, _, _, _ = items_to_judge[i]
                # parse output
                result = _parse_llm_json_output(response)
                # determine if success or not
                if "Failed to parse" not in result["reason"]:
                    yield original_index, result
                else:
                    failed_items_for_next_round.append(items_to_judge[i])
            
            items_to_judge = failed_items_for_next_round
            if items_to_judge:
                print(f"⚠️ {len(items_to_judge)} items failed parsing. Preparing for retry {attempt + 1}/{self.max_retries}...")
        
        for original_index, _, _ in items_to_judge:
            final_error_result = {
                "is_correct": False,
                "reason": f"Failed to parse model judgement after {self.max_retries} retries."
            }
            yield original_index, final_error_result