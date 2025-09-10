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

    # --- Judgement Options ---
    judge_group = parser.add_argument_group('Judgement Options', 'Configure a model-based judge.')
    judge_group.add_argument('--judge-type', type=str, choices=['api', 'local'],
                             help='Type of judge model. If not set, uses rule-based matching.')
    judge_group.add_argument('--judge-model-path', type=str,
                             help='Path to local judge model (for --judge-type local).')
    judge_group.add_argument('--tensor-parallel-size', type=int, help='Tensor parallel size for vllm.')
    judge_group.add_argument('--judge-api-endpoint', type=str,
                             help='API endpoint for judge model (for --judge-type api).')
    judge_group.add_argument('--judge-api-key', type=str, help='API key (optional).')
    judge_group.add_argument('--engine', type=str, choices=['vllm', 'hf'], default='vllm',
                             help='Inference engine for local judge (default: vllm).')
    judge_group.add_argument('--judge-model-name', type=str, help='Model name for API judge')
    
    args = parser.parse_args()

    judge_config: Optional[Dict[str, Any]] = None
    if args.judge_type:
        if args.judge_type == 'local' and not args.judge_model_path:
            parser.error("--judge-model-path is required for --judge-type local")
        
        judge_config = {
            "judge_type": args.judge_type, "model_path": args.judge_model_path, 'tensor_parallel_size': args.tensor_parallel_size,
            "api_endpoint": args.judge_api_endpoint, "api_key": args.judge_api_key,
            "engine": args.engine, "model_name": args.judge_model_name
        }
        if not args.quiet: print(f"⚖️  Using '{args.judge_type}' judge.")
    else:
        if not args.quiet: print("⚖️  Using default rule-based matching.")

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
    # ------------------- 核心修改在这里 -------------------
    import multiprocessing
    
    # 在Linux上，默认是 'fork'，这在多GPU的CUDA环境中是致命的。
    # 我们必须在任何CUDA/PyTorch/VLLM操作之前，
    # 强制将进程启动方法设置为 'spawn'。
    # "force=True" 是为了确保即使它已经被设置过，我们也能覆盖它。
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # 如果启动方法已经被设置，可能会抛出 RuntimeError。
        # 我们可以安全地忽略它，因为它可能已经被正确设置了。
        pass
    # ----------------------------------------------------
    main()