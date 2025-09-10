# src/evaluation/inference_engine.py

from abc import ABC, abstractmethod
from typing import List

class BaseInferenceEngine(ABC):
    """
    Abstract base class for inference engines.
    Defines a common interface for generating text from prompts.
    """
    @abstractmethod
    def __init__(self, model_path: str, **kwargs):
        pass

    @abstractmethod
    def batch_generate(self, prompts: List[str]) -> List[str]:
        """
        Takes a list of prompts and returns a list of generated texts.
        """
        pass

class VllmEngine(BaseInferenceEngine):
    """Inference engine using vLLM for high-throughput generation."""
    def __init__(self, model_path: str, tensor_parallel_size: int = 1):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vLLM is not installed. Please install it with 'pip install vllm'.")
        
        print(f"🚀 Initializing vLLM engine for model: {model_path}")
        # trust_remote_code is often needed for new models
        self.llm = LLM(model=model_path, trust_remote_code=True, tensor_parallel_size=tensor_parallel_size)
        # You can customize sampling params here
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)

    def batch_generate(self, prompts: List[str]) -> List[str]:
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]

class HfPipelineEngine(BaseInferenceEngine):
    """
    Inference engine using Hugging Face's transformers.pipeline.
    Slower, but good for compatibility.
    """
    def __init__(self, model_path: str, **kwargs):
        from transformers import pipeline, AutoTokenizer
        import torch

        print(f"🐌 Initializing Hugging Face pipeline engine for model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pipeline = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            **kwargs,
        )

    def batch_generate(self, prompts: List[str]) -> List[str]:
        # The pipeline is not great at batching, but we do our best.
        # It's important to control the output length and not generate the prompt.
        outputs = self.pipeline(
            prompts,
            max_new_tokens=1024,
            num_return_sequences=1,
            do_sample=False, # Use greedy decoding for consistency
            eos_token_id=self.tokenizer.eos_token_id,
        )
        # The output format is a list of lists, so we flatten and clean it.
        return [output[0]['generated_text'][len(prompt):] for prompt, output in zip(prompts, outputs)]
