import os
import argparse
import json
import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from typing import (
    Any,
    List,
    Dict,
    Optional,
    Union,
)

from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

# 导入 Bagel 的相关模块
from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae

# 从你的项目导入 inferencer
from inferencer import InterleaveInferencer

# =======================================
# 1. 模型加载函数 (与之前版本相同)
# =======================================
def load_bagel_model(model_path: str, max_mem_per_gpu: str = "40GiB"):
    """
    加载 Bagel 模型和相关的组件（Tokenizer, VAE, Transforms）。
    """
    print("="*20 + "  模型初始化开始  " + "="*20)
    
    # LLM config preparing
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # ViT config preparing
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    # VAE loading
    files_in_dir = os.listdir(model_path)
    print(f"路径 '{model_path}' 下找到的文件/文件夹列表: {files_in_dir}")
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    # Bagel config preparing
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model      = SiglipVisionModel(vit_config)
        model          = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    # Tokenizer Preparing
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # Image Transform Preparing
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    print("开始计算设备映射 (Device Map)...")
    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    print("Device Map:", device_map)

    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]

    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            device_map[k] = first_device
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
    
    print("开始加载模型权重并分发到多张 GPU...")
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_path, "ema.safetensors"),
        device_map=device_map,
        offload_buffers=False,
        dtype=torch.bfloat16,
        force_hooks=True,
        # offload_folder="/tmp/bagel_offload"
    )
    
    # ===============================================
    # 使用 PyTorch 自带函数打印显存
    # ===============================================
    print("\n" + "="*20 + "  PyTorch 显存报告  " + "="*20)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3) # PyTorch 预留的总显存
            print(f"  - GPU {i}: "
                  f"Allocated: {allocated:.2f} GB, "
                  f"Reserved: {reserved:.2f} GB")
    print("="*58 + "\n")
    # ===============================================

    model = model.eval()
    print("模型加载完成！")
    print("="*20 + "   模型初始化结束  " + "="*20)
    
    # 创建 Inferencer 实例
    inferencer = InterleaveInferencer(
        model=model, 
        vae_model=vae_model, 
        tokenizer=tokenizer, 
        vae_transform=vae_transform, 
        vit_transform=vit_transform, 
        new_token_ids=new_token_ids
    )
    
    return inferencer


# =======================================
# 2. 批量推理处理函数 (大改版)
# =======================================
def process_batch(question_file: str, output_dir: str, model_name: str, inferencer: 'InterleaveInferencer', batch_size: int):
    """
    批量处理 JSONL 文件中的问题。
    此版本会根据输入中图片数量对问题进行分组，并在每个批次处理后，
    使用缓冲机制立即尝试按原始顺序将结果写入文件。
    """
    # 1. 读取并过滤已处理的问题
    with open(question_file, 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f]

    os.makedirs(output_dir, exist_ok=True)
    
    input_file_name = os.path.splitext(os.path.basename(question_file))[0]
    output_filename = f"{input_file_name}_{model_name}.jsonl"
    output_file = os.path.join(output_dir, output_filename)
    
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                processed_ids.add(json.loads(line)['id'])
        print(f"找到 {len(processed_ids)} 个已处理的问题，将跳过它们。")
    
    questions_to_process = [q for q in questions if q.get("id") not in processed_ids]
    if not questions_to_process:
        print("没有需要处理的新问题。")
        return

    # 2. 按图片数量对问题进行分组，并记录原始索引 (逻辑不变)
    grouped_questions: Dict[int, List[Dict[str, Any]]] = {}
    for original_index, question in enumerate(questions_to_process):
        num_images = len(question.get("images", []))
        if num_images not in grouped_questions:
            grouped_questions[num_images] = []
        grouped_questions[num_images].append({
            "original_index": original_index,
            "data": question
        })

    # 3. (核心修改) 初始化缓冲区和写入指针
    results_buffer: Dict[int, Dict[str, Any]] = {}
    next_question_to_write_idx = 0
    
    # 定义推理超参数
    inference_hyper = dict(
        think=True,
        understanding_output=True,
        max_think_token_n=2048,
        do_sample=False,
        text_temperature=0.3,
    )

    print(f"开始处理 {len(questions_to_process)} 个问题，分为 {len(grouped_questions)} 个结构组。")
    
    # =============================================================
    # 获取总显存，用于计算百分比
    try:
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"检测到 GPU 总显存: {total_memory_gb:.2f} GB")
    except Exception as e:
        print(f"无法获取 GPU 信息: {e}")
        total_memory_gb = 0
    # =============================================================
    
    
    # 4. (核心修改) 打开文件一次，并在整个处理过程中保持打开状态
    with open(output_file, 'a', encoding='utf-8') as ans_file:
        with tqdm(total=len(questions_to_process), desc="Processing Questions") as progress_bar:
            # 遍历每个分组，对组内数据进行批处理
            for num_images, items_in_group in grouped_questions.items():
                
                for i in range(0, len(items_in_group), batch_size):
                    batch_items = items_in_group[i:i + batch_size]
                    
                    batch_input_lists = []
                    valid_batch_items = []
                    
                    for item in batch_items:
                        question_item = item["data"]
                        try:
                            input_list = [Image.open(p) for p in question_item.get("images", [])]
                            input_list.append(question_item.get("input_prompt", ""))
                            batch_input_lists.append(input_list)
                            valid_batch_items.append(item)
                        except Exception as e:
                            print(f"错误：跳过问题 ID {question_item.get('id')}，无法加载图片: {e}")
                            error_item = question_item.copy()
                            error_item["answer"] = f"LOAD_ERROR: {e}"
                            # 将错误结果放入缓冲区
                            results_buffer[item["original_index"]] = error_item
                            progress_bar.update(1)

                    if not batch_input_lists:
                        continue

                    # ==================== 显存监控开始 ====================
                    # 在推理前，重置峰值统计
                    torch.cuda.reset_peak_memory_stats() 
                    
                    # 记录推理前的显存占用
                    start_memory_gb = torch.cuda.memory_allocated() / (1024**3)
                    # =======================================================

                    # 调用批量推理方法
                    try:
                        batch_outputs = inferencer.batch_interleave_inference(
                            batch_input_lists=batch_input_lists, 
                            **inference_hyper
                        )
                        
                        # ==================== 显存监控报告 ===============
                        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
                        print(
                            f"\n[批次成功] "
                            f"结构: {num_images} 张图片, "
                            f"批次大小: {len(batch_input_lists)}. "
                            f"峰值显存 (张量): {peak_memory_gb:.2f} GB. "
                            f"增量: {peak_memory_gb - start_memory_gb:.2f} GB."
                        )
                        if total_memory_gb > 0:
                            print(f" -> 占总显存的 {peak_memory_gb / total_memory_gb:.1%}")
                        # ================================================
                        

                        for item, model_answer in zip(valid_batch_items, batch_outputs):
                            output_data = item["data"].copy()
                            output_data["answer"] = model_answer
                            # 将成功结果放入缓冲区
                            results_buffer[item["original_index"]] = output_data
                            
                    except torch.cuda.OutOfMemoryError:
                        print(
                            f"\n[显存溢出 OOM!] "
                            f"结构: {num_images} 张图片, "
                            f"批次大小: {len(batch_input_lists)}. "
                            f"请减小 batch_size！"
                        )
                        # 这里可以决定是退出程序还是跳过这个批次
                        raise  # 重新抛出异常，终止程序

                    except Exception as e:
                        print(f"模型批量推理时发生严重错误 (批次大小: {len(valid_batch_items)}, 图片数: {num_images}): {e}")
                        for item in valid_batch_items:
                            error_item = item["data"].copy()
                            error_item["answer"] = f"BATCH_INFERENCE_ERROR: {e}"
                            # 将推理错误结果放入缓冲区
                            results_buffer[item["original_index"]] = error_item
                    
                    # (核心修改) 处理完一个批次后，检查缓冲区并尝试写入文件
                    while next_question_to_write_idx in results_buffer:
                        result_to_write = results_buffer.pop(next_question_to_write_idx)
                        ans_file.write(json.dumps(result_to_write, ensure_ascii=False) + "\n")
                        next_question_to_write_idx += 1
                    
                    ans_file.flush() # 确保立即写入磁盘
                    progress_bar.update(len(batch_items))
                    
                    # ========== delete garbage of last batch ========
                    del batch_outputs
                    del batch_input_lists
                    torch.cuda.empty_cache()

    # 结束后，缓冲区中不应再有任何内容
    if results_buffer:
        print(f"警告：处理结束后，缓冲区中仍有 {len(results_buffer)} 个项目未写入。这可能表示处理逻辑有误。")

    print(f"处理完成！结果已保存至 {output_file}")




# =======================================
# 3. 主函数 (与之前版本相同)
# =======================================
def main():
    parser = argparse.ArgumentParser(description="使用 Bagel 模型批量处理多模态问题。")
    parser.add_argument('--checkpoint_dir', type=str, required=True, help="Bagel 模型所在的文件夹路径。")
    parser.add_argument('--question_file', type=str, required=True, help="包含问题的 JSONL 文件路径。")
    parser.add_argument('--output_dir', type=str, required=True, help="保存输出结果的目录。")
    parser.add_argument('--max_mem_per_gpu', type=str, default="40GiB", help="为每块 GPU 分配的最大显存。")
    parser.add_argument('--batch_size', type=int, default=4)
    
    args = parser.parse_args()

    # 设置随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 1. 加载模型
    inferencer = load_bagel_model(args.checkpoint_dir, args.max_mem_per_gpu)
    
    # 2. 从模型路径中提取模型名称，用于命名输出文件
    model_name = os.path.basename(args.checkpoint_dir.strip('/'))

    # 3. 开始批量推理
    process_batch(
        question_file=args.question_file,
        output_dir=args.output_dir,
        model_name=model_name,
        inferencer=inferencer,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    import torch.multiprocessing as mp
    # 尝试将多进程启动方法设置为 'spawn'
    # 必须在任何 CUDA 操作之前调用
    try:
        mp.set_start_method('spawn', force=True)
        print("INFO: 多进程启动方法已成功设置为 'spawn'。")
    except RuntimeError as e:
        print(f"WARN: 设置 'spawn' 失败: {e}。如果程序卡住，这可能是原因。")
    
    main()