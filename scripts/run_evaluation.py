# run_evaluation.py (精简后)
import sys
import os
import argparse
from typing import Dict, Any, Optional

current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)



def main():
    from evaluation.evaluator import batch_evaluate, BasicEvaluator, create_judge
    parser = argparse.ArgumentParser(
        description='Evaluation Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Rule-based evaluation on a single file
  python run_evaluation.py -i responses.jsonl
  
  # Batch evaluation with a local judge
  python run_evaluation.py -b results/ -o analysis/ --judge-type local --judge-model-path /path/to/model
        """
    )
    
    # --- Main action arguments ---
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', '-i', type=str, help='Input JSONL file path')
    group.add_argument('--batch_dir', '-b', type=str, help='Directory for batch evaluation')
    
    # --- Output arguments ---
    parser.add_argument('--output', '-o', type=str, help='Output file path for single evaluation')
    parser.add_argument('--output_dir', type=str, help='Output directory for batch evaluation')
    
    # --- Simplified Options ---
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')

    parser.add_argument('--judge-type', type=str, choices=['api', 'local'],
                             help='Type of judge model.')
    # --- Judgement Options ---
    local_group = parser.add_argument_group('Local Judgement Options', 'Configure a local judge.')

    # select local judge inference engine
    local_group.add_argument('--engine', type=str, choices=['vllm', 'hf'], default='vllm', help='Inference engine for local judge (default: vllm).')

    # --- configuration for vllm ---
    local_group.add_argument('--model-path', type=str,
                             help='Path to local judge model (for --judge-type local).')
    local_group.add_argument('--tensor-parallel-size', type=int, help='Tensor parallel size for vllm.')
    local_group.add_argument('--max-model-len', type=int, help='Max model length')

    local_group.add_argument('--temperature', type=float, help='Temperature parameter for vllm sampling')
    local_group.add_argument('--max-tokens', type=int, help='Max tokens parameter for vllm sampling')
    local_group.add_argument('--top-k', type=int, help='Top-k parameter for vllm sampling')
    local_group.add_argument('--top-p', type=float, help='Top-p parameter for vllm sampling')
    local_group.add_argument('--repetition-penalty', type=float, help='Repetition penalty')
    
    
    # --- configuration for api ---
    api_group = parser.add_argument_group('API Judgement Options', 'Configure a api judge')
    api_group.add_argument('--endpoint', type=str,
                             help='API endpoint for judge model (for --judge-type api).')
    api_group.add_argument('--api-key', type=str, help='API key.')
    api_group.add_argument('--timeout', type=int, default=120, help='Timeout for API request')
    api_group.add_argument('--model-name', type=str, help='Model for API request')
    api_group.add_argument('--max-retries', type=int, default=10, help='Max retry time for model, if the json result can\'t be parsed.')
    api_group.add_argument('--requests-per-minute', type=int, default=60, help='Requests sent per minute')
    
    args = parser.parse_args()

    judge_config: Optional[Dict[str, Any]] = None

    if args.judge_type == 'local':
        if not args.model_path:
            parser.error("--model-path is required for local model as judge. (For local model as judge, this should be the path to the model you use)")

        if args.engine == "vllm":
            model_config = {
                "model": args.model_path,
                "tensor_parallel_size": args.tensor_parallel_size,
                "max_model_len": args.max_model_len,
                "trust_remote_code": True
            }

            sampling_config = {
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
            vllm_config = {
                "model_config": model_config,
                "sampling_config": sampling_config,
            }
            judge_config = {
                "judge_type": args.judge_type, "vllm_config": vllm_config,
                "engine": args.engine
            }
        else:
            raise NotImplementedError("Not support other engines now")

    elif args.judge_type == 'api':
        api_config = {
            "endpoint": args.endpoint,
            "api_key": args.api_key,
            "timeout": args.timeout,
            "model": args.model_name,
            "max_retries": args.max_retries,
            "requests_per_minute": args.requests_per_minute,
        }
        judge_config = {
            "judge_type": args.judge_type,
            "api_config": api_config,
        }
    else:
        raise ValueError("Judge type must be either api or local")
    
    
    if not args.quiet: print(f"⚖️  Using '{args.judge_type}' judge.")

    try:
        if args.batch_dir:
            success = batch_evaluate(args.batch_dir, args.output_dir, judge_config=judge_config)
            if not success:
                sys.exit(1)
            
        elif args.input:
            if not os.path.exists(args.input):
                print(f"❌ Error: Input file '{args.input}' not found")
                sys.exit(1)

            # For single file evaluation, we create the judge and evaluator here.
            judge = create_judge(judge_config)
            evaluator = BasicEvaluator(judge=judge)
            evaluator.evaluate(args.input, args.output)
            if args.output:
                print(f"📁 Results saved to {args.output}")

    except Exception as e:
        print(f"❌ An unhandled error occurred: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()