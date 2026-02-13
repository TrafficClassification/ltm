export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

python evaluation.py \
  --model_name /media/inspur/disk/sqkang_workspace/projects/trafficllm/qwen3_4b \
  --output_dir /media/inspur/disk/sqkang_workspace/projects/trafficllm/train_output \
  --lora_weights /media/inspur/disk/sqkang_workspace/projects/trafficllm/output_vpn_cls6_6pac_8epochs/final_weights \
  --dataset_test /media/inspur/disk/sqkang_workspace/projects/trafficllm/dataset/iscx_vpn_cls6_4pac_less_test.jsonl \
  --results results.jsonl \
  --batch_size 4 \
  --max_length 1024 \