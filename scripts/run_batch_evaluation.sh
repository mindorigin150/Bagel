#!/bin/bash
set -e

# --- check parameters ---
if [ $# -eq 0 ]; then
    echo "❌ Error: Please provide an input directory."
    echo "Usage: bash $0 <input_directory> [judge_options...]"
    exit 1
fi

INPUT_DIR="$1"
PYTHON_ARGS="${@:2}"
BASE_OUTPUT_DIR="./data/evaluate"

INPUT_DIR_BASENAME=$(basename "$(realpath -s "${INPUT_DIR}")")
FINAL_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${INPUT_DIR_BASENAME}"
mkdir -p "${FINAL_OUTPUT_DIR}"

echo "🚀 Starting evaluation task..."
echo "  Input Directory: ${INPUT_DIR}"
echo "  Final Output Directory: ${FINAL_OUTPUT_DIR}"
echo "  Judge Arguments: ${PYTHON_ARGS}"

python -u -m ipdb -c continue scripts/run_evaluation.py \
    --batch_dir "${INPUT_DIR}" \
    --output_dir "${FINAL_OUTPUT_DIR}" \
    ${PYTHON_ARGS}

echo "✅ Batch evaluation script finished. Results are in ${FINAL_OUTPUT_DIR}"
