export CUDA_VISIBLE_DEVICES=1
python predict.py \
  --model_name_or_path /home/xjtu/workspace/ltm/models/Qwen3_17b \
  --lora_model_path /home/xjtu/workspace/ltm/output/first_n/pac4/ustc_malware_1token_17b/final_weights \
  --dataset_unlabeled /home/xjtu/workspace/ltm/dataset/mixture_study/ustc_malware_1token_mixture_test.jsonl \
  --output_file /home/xjtu/workspace/ltm/output/mixture_study/17b/ustc_malware_predict_results.jsonl \
  --max_new_tokens 1 \
  --batch_size 32 \
  --bf16