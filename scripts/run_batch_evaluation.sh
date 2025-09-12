#!/bin/bash
set -e

# --- 1. 检查基本参数 ---
if [ $# -eq 0 ]; then
    echo "❌ 错误: 请提供输入数据目录。"
    echo "用法: bash $0 <输入目录> [评测参数...]"
    exit 1
fi

INPUT_DIR="$1"
PYTHON_ARGS="${@:2}" # 捕获除第一个参数外的所有参数
BASE_OUTPUT_DIR="./data/evaluate"

# --- 2. 智能解析模型名称 ---
#    - 如果找到 --model-path, 则取其路径的最后一部分 (basename) 作为模型名
#    - 如果找到 --model-name, 则直接使用其值作为模型名
MODEL_NAME=""
# 遍历所有传入的参数
for ((i=1; i<=$#; i++)); do
    # 检查当前参数是否是 '--model-path'
    if [ "${!i}" == "--model-path" ]; then
        # 它后面的一个参数就是模型路径
        next_i=$((i+1))
        # 使用 basename 命令从路径中提取最后的文件夹名
        MODEL_NAME=$(basename "${!next_i}")
        break # 找到后就跳出循环

    # 检查当前参数是否是 '--model-name'
    elif [ "${!i}" == "--model-name" ]; then
        # 它后面的一个参数就是模型名称
        next_i=$((i+1))
        # 直接使用该值作为模型名
        MODEL_NAME="${!next_i}"
        break # 找到后就跳出循环
    fi
done

# 如果循环结束后，MODEL_NAME 仍然为空，说明两个参数都未提供
if [ -z "${MODEL_NAME}" ]; then
    echo "❌ 错误: 未在参数中找到 --model-path 或 --model-name。无法确定模型名称。"
    exit 1
fi

# --- 3. 构建新的、更详细的输出目录 ---
INPUT_DIR_BASENAME=$(basename "$(realpath -s "${INPUT_DIR}")")
# 拼接最终的输出目录：./data/evaluate/输入数据目录名/模型名
FINAL_OUTPUT_DIR="${BASE_OUTPUT_DIR}_${MODEL_NAME}/${INPUT_DIR_BASENAME}"
mkdir -p "${FINAL_OUTPUT_DIR}"

echo "🚀 开始评测任务..."
echo "  - 输入目录: ${INPUT_DIR}"
echo "  - 解析出的模型名称: ${MODEL_NAME}"
echo "  - 最终输出目录: ${FINAL_OUTPUT_DIR}"
echo "  - 传递给Python的参数: ${PYTHON_ARGS}"

# --- 4. 执行Python评测脚本 ---
# 将新的FINAL_OUTPUT_DIR传递给 --output_dir
python -u -m ipdb -c continue scripts/run_evaluation.py \
    --batch_dir "${INPUT_DIR}" \
    --output_dir "${FINAL_OUTPUT_DIR}" \
    ${PYTHON_ARGS}

echo "✅ 批量评测脚本执行完毕。结果已保存至 ${FINAL_OUTPUT_DIR}"
