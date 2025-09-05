# Examples of batch infer

```python
python batch_infer_understanding.py \
    --checkpoint_dir "/run/determined/NAS1/public/HuggingFace/BAGEL-7B-MoT" \
    --question_file "${DATA_PATH}/data/prompts/general/ff_rsn/MindCube_tinybench_ff_rsn_0_interval.jsonl" \
    --output_dir "${DATA_PATH}/data/results/frozen_bagel/ff_rsn" \
    --batch_size 32
```