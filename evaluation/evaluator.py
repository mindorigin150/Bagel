import os
import glob
from typing import Dict, Optional, Any
from torch import Value
from tqdm import tqdm

from .judgement import BaseJudge, LocalJudge, APIModelJudge, JudgementError
from .inference_engine import VllmEngine, HfPipelineEngine

from .utils import (
    get_setting_from_id,
    initialize_basic_results_structure,
    update_accuracy_metrics,
    load_jsonl_data,
    save_json_results,
    print_basic_results
)

class BasicEvaluator:
    """ 
    Basic task evaluator for answer accuracy.
    Refactored to use batch judging for efficiency.
    """
    
    def __init__(self, judge: BaseJudge):
        self.judge = judge
        print(f"BasicEvaluator initialized with judge: {self.judge.__class__.__name__}")
    
    def evaluate(self, jsonl_path: str, output_path: Optional[str] = None) -> Dict:
        data = load_jsonl_data(jsonl_path)
        
        results = initialize_basic_results_structure()
        results['unfiltered_total'] = len(data)
        error_cases = {'extraction_error': [], 'judgement_error': []}

        # ---- 1. collect all assessment tasks ----
        items_to_judge = []
        original_indices = [] # Keep track of where each item came from.
        for i, item in enumerate(data):
            if item.get('gt_answer'):
                model_answer = item.get('answer', '')
                gt_answer = item.get('gt_answer', '')
                items_to_judge.append((model_answer, gt_answer))
                original_indices.append(i)

        # ---- 2. batch_process ----
        judgements = []
            # use judge to process answers
        print(f"⚖️  Submitting {len(items_to_judge)} items to judge in a single batch...")
        try:
            judgements = self.judge.batch_judge(items_to_judge)
        except JudgementError as e:
            # If batch fails, we can't continue.
            print(f"\n❌ FATAL: Batch judgement failed. Error: {e}")
            error_cases['judgement_error'].append({'id': 'BATCH_FAILED', 'error': str(e)})
            return {'results': results, 'error_cases': error_cases, 'processed_data': data}

        # ---- 3. integrate ----
        # Loop over in-memory data. This is fast.
        filtered_total = 0
        filtered_correct = 0
        for original_idx, judgement in zip(original_indices, judgements):
            item = data[original_idx]
            is_correct = judgement['is_correct']
            
            item['judgement_is_correct'] = is_correct
            item['judgement_reason'] = judgement['reason']
            item['judgement_method'] = self.judge.__class__.__name__ if self.judge else "RuleBased"

            # Your original metric calculation logic is fine.
            setting = get_setting_from_id(item.get('id', ''))
            results['settings'][setting]['total'] += 1
            include_in_overall = results['settings'][setting].get('include_in_overall', True)
            if include_in_overall:
                filtered_total += 1
            
            if is_correct:
                results['settings'][setting]['gen_cogmap_correct'] += 1
                if include_in_overall:
                    filtered_correct += 1

        results['total'] = filtered_total
        results['gen_cogmap_correct'] = filtered_correct
        results = update_accuracy_metrics(results)
        
        final_results = {'results': results, 'error_cases': error_cases, 'processed_data': data}
        
        print_basic_results(results)
        if output_path:
            save_json_results(final_results, output_path)
        
        return final_results


def create_judge(judge_config: Optional[Dict[str, Any]]) -> Optional[BaseJudge]:
    """Factory function to create a judge instance based on config."""
    if not judge_config:
        return None
    
    judge_type = judge_config.get("judge_type")
    # API judge
    if judge_type == "api":
        endpoint = judge_config.get("api_endpoint")
        if not endpoint: raise ValueError("api_endpoint must be provided for API judge.")
        api_key = judge_config.get("api_key")
        if not api_key: raise ValueError("api_key must be provided for API judge.")
        model_name = judge_config.get("model_name")
        if not model_name: raise ValueError("model_name must be provided for API judge.")
        return APIModelJudge(endpoint=endpoint, api_key=api_key, model=model_name)
    # local model judge
    elif judge_type == "local":
        model_path = judge_config.get("model_path")
        if not model_path: raise ValueError("model_path must be provided for local judge.")
        
        engine_type = judge_config.get("engine", "vllm")
        tensor_parallel_size = judge_config.get("tensor_parallel_size", 1)
        print(f"🔧 Creating '{engine_type}' engine for local judge...")
        
        if engine_type == "vllm": engine = VllmEngine(model_path, tensor_parallel_size)
        elif engine_type == "hf": engine = HfPipelineEngine(model_path)
        else: raise ValueError(f"Unknown engine type: {engine_type}")
        
        return LocalJudge(engine)
    else:
        raise ValueError(f"Unsupported judge type: {judge_type}")


def batch_evaluate(directory: str, output_dir: Optional[str] = None, 
                   judge_config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Efficiently evaluates all .jsonl files in a directory using BasicEvaluator.
    The judge model is loaded only ONCE.
    """
    jsonl_files = sorted(glob.glob(os.path.join(directory, '**', '*.jsonl'), recursive=True))
    if not jsonl_files:
        print(f"⚠️ No .jsonl files found in {directory}. Nothing to do.")
        return True

    try:
        judge = create_judge(judge_config)
    except Exception as e:
        print(f"❌ FATAL: Failed to create the judge. Aborting. Error: {e}")
        return False

    print(f"🔍 Found {len(jsonl_files)} files to evaluate.")
    failed_files = []
    
    # Create the evaluator once with the pre-loaded judge.
    evaluator = BasicEvaluator(judge=judge)
    
    for i, file_path in enumerate(jsonl_files):
        print(f"\n--- [{i+1}/{len(jsonl_files)}] Processing: {os.path.basename(file_path)} ---")
        output_path = None
        if output_dir:
            relative_path = os.path.relpath(file_path, directory)
            base, _ = os.path.splitext(relative_path)
            output_path = os.path.join(output_dir, f"{base}_eval_results.json")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            evaluator.evaluate(file_path, output_path)
            print(f"✅ Successfully evaluated {os.path.basename(file_path)}")
        except Exception as e:
            print(f"❌ FAILED to evaluate {os.path.basename(file_path)}. Error: {e}")
            failed_files.append(file_path)

    if failed_files:
        print(f"\n🚨 Batch evaluation completed with {len(failed_files)} failures:")
        for f in failed_files: print(f"  - {f}")
        return False
    
    print("\n🎉 All files processed successfully!")
    return True
