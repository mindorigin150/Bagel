## run_batch_evaluation guide

For run local model as local judge, run the following:
```bash
bash scripts/run_batch_evaluation.sh /your_data_path \
--judge-type local \
--judge-model-path /your_model_path \
--tensor-parallel-size num_of_gpus
```

For calling internvl3.5-241b-a28b using API, run the following

```bash
bash scripts/run_batch_evaluation.sh /your_data_path \
--judge-type api \
--judge-api-endpoint https://chat.intern-ai.org.cn/api/v1/chat/completions \
--judge-api-key sk-xxx \
--judge-model-name internvl3.5-241b-a28b
```