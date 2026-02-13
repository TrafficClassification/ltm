import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import DataCollatorForLanguageModeling,DataCollatorForSeq2Seq
import functools
import json
import torch.distributed as dist
from datetime import datetime



def print_rank_0(msg):
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    print(msg)


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune Qwen3 with LoRA')
    
    parser.add_argument('--model_name', type=str, default="",
                        help='Path to the base model')

    parser.add_argument('--dataset_test', type=str, default="",
                        help='Path to the test dataset (JSON format)')
    parser.add_argument('--lora_weights', type=str, default="",
                        help='Path to the LoRA weights')
    

    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size per GPU/TPU core during training')

    parser.add_argument('--max_length', type=int, default=384,
                        help='Maximum sequence length')
    parser.add_argument('--results', type=str, default="results.jsonl",
                        help='Path to save evaluation results')
    parser.add_argument('--output_dir', type=str, default="",
                        help='Directory to save model checkpoints')
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

# ======================
# 1. 解析命令行参数
# ======================
args = parse_args()

# ======================
# 2. 配置参数（使用解析后的参数）
# ======================
MODEL_NAME = args.model_name
LORA_WEIGHTS = args.lora_weights
RESULTS = args.results
OUTPUT_DIR = args.output_dir
BATCH_SIZE = args.batch_size
MAX_LENGTH = args.max_length
SEED = args.seed

# ======================
# 3. 加载模型和tokenizer
# ======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto"

)
peft_model = PeftModel.from_pretrained(model, LORA_WEIGHTS) 

merged_model = peft_model.merge_and_unload()  # 关键：合并后卸载适配器
merged_model.eval()

def preprocess_function(examples):
    inputs = []
    labels = []
    
    for messages in examples["messages"]:
        try:
            # 提取对话内容（添加异常处理，避免角色缺失导致崩溃）
            system_content = next(m["content"] for m in messages if m["role"] == "system")
            user_content = next(m["content"] for m in messages if m["role"] == "user")
            assistant_content = next(m["content"] for m in messages if m["role"] == "assistant")
        except StopIteration:

            continue
        

        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking = False
        )
        full_prompt = f"{prompt}{assistant_content}"

        tokenized_full = tokenizer(
            full_prompt,
            max_length=MAX_LENGTH,
            truncation=True,
            #padding="max_length",
            #add_special_tokens=True,
            return_tensors=None
        )
        input_ids = tokenized_full["input_ids"]  
        #print_rank_0(f"full_prompt: {input_ids} \n")
        if len(input_ids)==MAX_LENGTH:
            print("invalid length")
        tokenized_prompt = tokenizer(
            prompt,
            max_length=MAX_LENGTH,  
            truncation=True,        
            add_special_tokens=False,
            return_tensors=None
        )
        #print_rank_0(f"prompt: {input_ids} \n")
        prompt_len = len(tokenized_prompt["input_ids"])  # 截断后的prompt长度
        

        label_ids = []

        label_ids.extend(input_ids[prompt_len:])
        #label_ids.append(151645)
        label_ids = [-100]*prompt_len+label_ids
        #input_ids = [tokenizer.pad_token_id]*(MAX_LENGTH-len(input_ids))+input_ids
        #print_rank_0(f"label_ids: {label_ids} \n")
        # 4. 校验长度（可选，用于调试）
        #assert len(input_ids) == MAX_LENGTH, f"input_ids长度错误: {len(input_ids)}"
        #assert len(label_ids) == MAX_LENGTH, f"label_ids长度错误: {len(label_ids)}"
        assert len(input_ids) == len(label_ids), f"input_ids长度错误: {len(input_ids)}"
        inputs.append(input_ids)
        labels.append(label_ids)

    return {"input_ids": inputs, "labels": labels}

test_dataset = load_dataset("json", data_files=args.dataset_test,split="train")     

# loading test dataset
tokenized_test_dataset = test_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=128,
    remove_columns=test_dataset.column_names
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_steps=100,
    do_train = False,
    do_eval = True,
    save_strategy="no",
    eval_accumulation_steps=16,
    bf16=True,
    load_best_model_at_end=False,
    seed=SEED
)


def preprocess_logits_for_metrics(logits, labels):
    #print(f"logits: {logits.shape}")
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids



def compute_metrics(eval_preds, tokenizer):
    #print_rank_0(f"eval_preds: {eval_preds.inputs} \n")
    all_pred_ids, all_label_ids = eval_preds
    
    correct = 0
    total = 0
    skipped = 0

    for i in range(len(all_pred_ids)):
        pred_ids = all_pred_ids[i]          # shape: [seq_len]
        label_ids = all_label_ids[i]        # shape: [seq_len], with -100 for prompt

        # 尝试移位操作
        pred_ids = pred_ids[:-1]
        label_ids = label_ids[1:]

        # 找出 label 的位置（非 -100 的位置）
        mask = label_ids != -100
        if not mask.any():
            skipped += 1
            continue
        
        # 提取真实标签 token IDs
        #print_rank_0(f"label_ids: {label_ids}")
        #print_rank_0(f"pred: {pred_ids}")
        valid_label_ids = label_ids[mask]           # shape: [L]
        # 提取模型在这些位置上的预测
        pred_for_label = pred_ids[mask]             # ✅ 关键：用相同 mask 取预测！
        #print_rank_0(f"pred_for_label: {pred_for_label}")
        #print_rank_0(f"valid_label_ids: {valid_label_ids}")
        try:
            label_text = tokenizer.decode(valid_label_ids, skip_special_tokens=True).strip()
            pred_text = tokenizer.decode(pred_for_label, skip_special_tokens=True).strip()
            if pred_text == label_text:
                correct += 1
            #print_rank_0(f"Correct: {label_text} | Pred: {pred_text}")
            total += 1

        except Exception as e:
            skipped += 1
            continue

    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": round(accuracy, 6),
        "correct": correct,
        "total_valid": total,
        "skipped_samples": skipped,
        "total_samples": len(all_pred_ids)
    }

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=merged_model,
    label_pad_token_id=-100,          # 关键：padding 部分不参与 loss 计算
    padding="longest",                # 动态 padding 到 batch 最长
    pad_to_multiple_of=8,             # 提升 GPU 效率
)


trainer = Trainer(
    model=merged_model,
    args=training_args,
    eval_dataset=tokenized_test_dataset,
    data_collator=data_collator,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    compute_metrics=functools.partial(compute_metrics, tokenizer=tokenizer)
)


print_rank_0("Evaluating on test set...")
test_results = trainer.evaluate(eval_dataset=tokenized_test_dataset)
print_rank_0(f"Test Metrics: {test_results}")

result_with_metadata = {
    **test_results,
    "timestamp": datetime.now().isoformat(),
    "model_name": MODEL_NAME,
    "lora_weights": LORA_WEIGHTS,
    "dataset_test": args.dataset_test,
    "batch_size": BATCH_SIZE,
    "max_length": MAX_LENGTH
}

with open(RESULTS, "a", encoding="utf-8") as f:
    f.write(json.dumps(result_with_metadata, ensure_ascii=False) + "\n")


if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()