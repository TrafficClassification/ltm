import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import DataCollatorForLanguageModeling,DataCollatorForSeq2Seq
import functools
import json
import torch.distributed as dist




def print_rank_0(msg):
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    print(msg)


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune Qwen3 with LoRA')
    
    # Model and dataset parameters
    parser.add_argument('--model_name', type=str, default="",
                        help='Path to the base model')
    parser.add_argument('--dataset_train', type=str, default="",
                        help='Path to the training dataset (JSON format)')
    parser.add_argument('--dataset_valid', type=str, default="",
                        help='Path to the validation dataset (JSON format)')
    parser.add_argument('--dataset_test', type=str, default="",
                        help='Path to the test dataset (JSON format)')
    
    parser.add_argument('--output_dir', type=str, default="",
                        help='Directory to save model checkpoints')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size per GPU/TPU core during training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Total number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Initial learning rate')
    parser.add_argument('--max_length', type=int, default=384,
                        help='Maximum sequence length')
    parser.add_argument('--lora_r', type=int, default=8,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                        help='LoRA dropout probability')
    
    # Optimization parameters
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help='Number of steps to accumulate gradients before updating weights')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--warmup_ratio', type=float, default=0.03,
                        help='Ratio of steps for warmup')
    
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

OUTPUT_DIR = args.output_dir
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
MAX_LENGTH = args.max_length
LORA_R = args.lora_r
LORA_ALPHA = args.lora_alpha
LORA_DROPOUT = args.lora_dropout
GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
WEIGHT_DECAY = args.weight_decay
WARMUP_RATIO = args.warmup_ratio
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

)

"""def preprocess_function(examples):
    inputs = []
    labels = []
    
    for messages in examples["messages"]:
        system_content = next(m["content"] for m in messages if m["role"] == "system")
        user_content = next(m["content"] for m in messages if m["role"] == "user")
        assistant_content = next(m["content"] for m in messages if m["role"] == "assistant")
        
        # 构建对话模板（Qwen3标准格式）
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            tokenize=False,
            add_generation_prompt=False
        )
        
        full_prompt = f"{prompt}{assistant_content}"
        
        tokenized = tokenizer(
            full_prompt,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = tokenized.input_ids[0].tolist()
        label_ids = [-100] * len(tokenizer(prompt, add_special_tokens=False).input_ids) + input_ids[len(tokenizer(prompt, add_special_tokens=False).input_ids):]

        inputs.append(input_ids)
        labels.append(label_ids)
    
    return {"input_ids": inputs, "labels": labels}
"""
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
train_dataset = load_dataset("json", data_files=args.dataset_train,split="train")
valid_dataset = load_dataset("json", data_files=args.dataset_valid,split="train")
test_dataset = load_dataset("json", data_files=args.dataset_test,split="train")     

# 对训练集进行预处理
tokenized_train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=128,
    remove_columns=train_dataset.column_names
)

# 对验证集进行预处理
tokenized_val_dataset = valid_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=128,
    remove_columns=valid_dataset.column_names
)
tokenized_test_dataset = test_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=128,
    remove_columns=test_dataset.column_names
)
# ======================
# 4. LoRA配置
# ======================
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ======================
# 5. 训练配置
# ======================
training_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=400,
    save_strategy="no",
    eval_strategy="steps",
    eval_steps=800,
    eval_accumulation_steps=16,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    bf16=True,
    load_best_model_at_end=False,
    ddp_find_unused_parameters=False,
    seed=SEED,
)
# ======================
# 6. 评估函数
# ======================


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


class CausalLMCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
    def __call__(self, features):
        # 提取input_ids、attention_mask、labels（确保每个样本都有这三个字段）
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features] if "attention_mask" in features[0] else None
        labels = [f["labels"] for f in features]

        # 对input_ids做padding（用pad_token_id填充）
        max_len = MAX_LENGTH #max(len(ids) for ids in input_ids)
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for idx in range(len(input_ids)):
            #label 偏移
            labels[idx] = labels[idx][1:]+[-100]
        
            # 处理input_ids
            pad_len = max_len - len(input_ids[idx])
            if self.tokenizer.padding_side == "left":
                padded_input = [self.pad_token_id] * pad_len + input_ids[idx]
            else:
                padded_input = input_ids[idx] + [self.pad_token_id] * pad_len
            padded_input_ids.append(padded_input)

            # 处理attention_mask（如果存在）
            if attention_mask is not None:
                if self.tokenizer.padding_side == "left":
                    padded_attention = [0] * pad_len + attention_mask[idx]  # 0表示不关注padding token
                else:
                    padded_attention = attention_mask[idx] + [0] * pad_len  # 0表示不关注padding token
                padded_attention_mask.append(padded_attention)

            # 处理labels（用-100填充，确保padding部分不参与损失计算）
            if self.tokenizer.padding_side == "left":
                padded_label = [-100] * pad_len + labels[idx]
            else:
                padded_label = labels[idx] + [-100] * pad_len
            padded_labels.append(padded_label)

        # 转换为torch张量
        result = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }
        if attention_mask is not None:
            result["attention_mask"] = torch.tensor(padded_attention_mask, dtype=torch.long)

        return result

# 初始化自定义Collator
#data_collator = CausalLMCollator(tokenizer=tokenizer)
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,          # 关键：padding 部分不参与 loss 计算
    padding="longest",                # 动态 padding 到 batch 最长
    pad_to_multiple_of=8,             # 提升 GPU 效率
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    compute_metrics=functools.partial(compute_metrics, tokenizer=tokenizer)
)


# ======================
# 8. 开始训练
# ======================
trainer.train()

lora_output_dir = os.path.join(OUTPUT_DIR,"final_weights")
os.makedirs(lora_output_dir, exist_ok=True)

model.save_pretrained(lora_output_dir, safe_serialization=True)

# ======================
# 10. 评估模型
# ======================

print_rank_0("Evaluating on test set...")
test_results = trainer.evaluate(eval_dataset=tokenized_test_dataset)
print_rank_0(f"Test Metrics: {test_results}")

with open(os.path.join(OUTPUT_DIR, "test_results.json"), "w") as f:
    json.dump(test_results, f, indent=4)

print_rank_0("\nTraining completed successfully!")
print_rank_0(f"Model saved to: {os.path.join(OUTPUT_DIR, 'final_model')}")
print_rank_0(f"Training results saved to: {os.path.join(OUTPUT_DIR, 'training_results.json')}")
if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()