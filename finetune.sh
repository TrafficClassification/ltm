export TOKENIZERS_PARALLELISM=false

accelerate launch finetune.py \
  --model_name /home/xjtu/workspace/ltm/models/Qwen3_17b \
  --dataset_train /home/xjtu/workspace/ltm/dataset/first_n/pac4/vpn_services_1token_train.jsonl \
  --dataset_valid /home/xjtu/workspace/ltm/dataset/first_n/pac4/vpn_services_1token_valid.jsonl \
  --dataset_test /home/xjtu/workspace/ltm/dataset/first_n/pac4/vpn_services_1token_test.jsonl \
  --output_dir /home/xjtu/workspace/ltm/output/first_n/pac4/vpn_services_1token_17b \
  --batch_size 1 \
  --epochs 8 \
  --learning_rate 2e-4 \
  --max_length 1024 \
  --gradient_accumulation_steps 16 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.1

accelerate launch finetune.py \
  --model_name /home/xjtu/workspace/ltm/models/Qwen3_17b \
  --dataset_train /home/xjtu/workspace/ltm/dataset/first_n/pac4/ustc_malware_1token_train.jsonl \
  --dataset_valid /home/xjtu/workspace/ltm/dataset/first_n/pac4/ustc_malware_1token_valid.jsonl \
  --dataset_test /home/xjtu/workspace/ltm/dataset/first_n/pac4/ustc_malware_1token_test.jsonl \
  --output_dir /home/xjtu/workspace/ltm/output/first_n/pac4/ustc_malware_1token_17b \
  --batch_size 1 \
  --epochs 2 \
  --learning_rate 2e-4 \
  --max_length 1024 \
  --gradient_accumulation_steps 16 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.1

accelerate launch finetune.py \
  --model_name /home/xjtu/workspace/ltm/models/Qwen3_17b \
  --dataset_train /home/xjtu/workspace/ltm/dataset/first_n/pac4/ustc_benign_1token_train.jsonl \
  --dataset_valid /home/xjtu/workspace/ltm/dataset/first_n/pac4/ustc_benign_1token_valid.jsonl \
  --dataset_test /home/xjtu/workspace/ltm/dataset/first_n/pac4/ustc_benign_1token_test.jsonl \
  --output_dir /home/xjtu/workspace/ltm/output/first_n/pac4/ustc_benign_1token_17b \
  --batch_size 1 \
  --epochs 2 \
  --learning_rate 2e-4 \
  --max_length 1024 \
  --gradient_accumulation_steps 16 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.1