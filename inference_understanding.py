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
# 2. 批量推理处理函数 (已修正)
# =======================================
def process_batch(question_file: str, output_dir: str, model_name: str, inferencer: InterleaveInferencer):
    """
    批量处理 JSONL 文件中的问题。
    """
    with open(question_file, 'r') as f:
        questions = [json.loads(line) for line in f]

    os.makedirs(output_dir, exist_ok=True)
    
    input_file_name = os.path.splitext(os.path.basename(question_file))[0]
    output_filename = f"{input_file_name}_{model_name}.jsonl"
    output_file = os.path.join(output_dir, output_filename)

    images_output_dir_name = f"{input_file_name}_{model_name}_images"
    images_output_dir = os.path.join(output_dir, images_output_dir_name)
    os.makedirs(images_output_dir, exist_ok=True)
    
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                processed_ids.add(json.loads(line)['id'])
        print(f"找到 {len(processed_ids)} 个已处理的问题，将跳过它们。")

    with open(output_file, 'a') as ans_file:
        for question_item in tqdm(questions, desc="Processing Questions"):
            q_id = question_item.get("id")

            if q_id in processed_ids:
                continue
                
            image_paths = question_item.get("images", [])
            prompt = question_item.get("input_prompt", "")
            
            # 【修正点】: 正确构建 interleave_inference 的输入列表
            # 格式为: [Image, Image, ..., str]
            input_list = []
            try:
                for path in image_paths:
                    input_list.append(Image.open(path))
                input_list.append(prompt)
            except Exception as e:
                print(f"错误：无法加载问题 ID {q_id} 的图片: {e}")
                continue

            # 定义推理超参数
            inference_hyper = dict(
                think=True,  # 强制模型进行思考，以输出 <think>...</think>
                understanding_output=True, # 这是理解任务，所以设置为 True
                max_think_token_n=1024,
                do_sample=False,
                text_temperature=0.3, # 在 do_sample=False 时不起作用，但保留
            )

            try:
                # 【修正点】: 直接调用 interleave_inference 而不是 __call__
                output_list = inferencer.interleave_inference(
                    input_lists=input_list, 
                    **inference_hyper
                )

                # 【修正点】: 从返回的列表中解析出文本和图片
                model_answer = None
                generated_images = []
                for item in output_list:
                    if isinstance(item, str):
                        model_answer = item
                    elif isinstance(item, Image.Image):
                        generated_images.append(item)
                
                # 如果没有解析到文本，则标记为错误
                if model_answer is None:
                    raise ValueError("模型没有生成任何文本输出。")

            except Exception as e:
                print(f"模型推理时发生错误 (ID: {q_id}): {e}")
                error_item = question_item.copy()
                error_item["answer"] = f"INFERENCE_ERROR: {e}"
                ans_file.write(json.dumps(error_item) + "\n")
                ans_file.flush()
                continue

            # 保存生成的中间图片
            if generated_images:
                item_image_dir = os.path.join(images_output_dir, str(q_id))
                os.makedirs(item_image_dir, exist_ok=True)
                for i, img in enumerate(generated_images):
                    img_path = os.path.join(item_image_dir, f"generated_{i}.png")
                    img.save(img_path)

            # 写入结果到 JSONL 文件
            output_data = question_item.copy()
            output_data["answer"] = model_answer
            
            ans_file.write(json.dumps(output_data) + "\n")
            ans_file.flush()
                
    print(f"处理完成！结果已保存至 {output_file}")
    if any(os.scandir(images_output_dir)): # 检查目录是否非空
        print(f"生成的中间图片已保存至 {images_output_dir}")
    else:
        # 如果没有生成任何图片，删除空目录
        try:
            os.rmdir(images_output_dir)
        except OSError:
            pass # 目录非空则不删除



# =======================================
# 3. 主函数 (与之前版本相同)
# =======================================
def main():
    parser = argparse.ArgumentParser(description="使用 Bagel 模型批量处理多模态问题。")
    parser.add_argument('--checkpoint_dir', type=str, required=True, help="Bagel 模型所在的文件夹路径。")
    parser.add_argument('--question_file', type=str, required=True, help="包含问题的 JSONL 文件路径。")
    parser.add_argument('--output_dir', type=str, required=True, help="保存输出结果的目录。")
    parser.add_argument('--max_mem_per_gpu', type=str, default="40GiB", help="为每块 GPU 分配的最大显存。")
    
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