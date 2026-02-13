import os
import json
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel

# ======================
# 参数解析
# ======================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--lora_model_path', type=str, required=True)
    parser.add_argument('--dataset_unlabeled', type=str, required=True)
    parser.add_argument('--output_file', type=str, default='predictions.jsonl')
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--max_new_tokens', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--load_in_4bit', action='store_true')
    return parser.parse_args()

# ======================
# 工具函数：构建 prompt（与 preprocess_function 一致）
# ======================
def build_prompt(system_content: str, user_content: str, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

# ======================
# 自定义 Dataset
# ======================
class UnlabeledDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.original_messages = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    messages = item["messages"]
                    self.original_messages.append(messages)

                    # === 关键修改：只取 system 和 user，忽略 assistant ===
                    system_content = ""
                    user_content = ""
                    for m in messages:
                        if m["role"] == "system":
                            system_content = m["content"]
                        elif m["role"] == "user":
                            user_content = m["content"]
                        # 注意：这里故意跳过 role=="assistant"

                    prompt = build_prompt(system_content, user_content, tokenizer)
                    self.data.append(prompt)

                except Exception as e:
                    print(f"Skipping invalid line: {line} | Error: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "prompt": self.data[idx],
            "original_messages": self.original_messages[idx]
        }

# ======================
# Collate Function
# ======================
def collate_fn(batch, tokenizer, max_length):
    prompts = [item["prompt"] for item in batch]
    original_msgs = [item["original_messages"] for item in batch]

    tokenized = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=False,
        #pad_to_multiple_of=8
    )
    return {
        "input_ids": tokenized.input_ids,
        "attention_mask": tokenized.attention_mask,
        "prompts": prompts,
        "original_messages": original_msgs
    }

# ======================
# 主函数
# ======================
def main():
    args = parse_args()

    # === 1. 加载 tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        padding_side='left'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === 2. 加载模型 ===
    print("Loading base model...")
    bnb_config = None
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )

    # === 3. 加载 LoRA ===
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, args.lora_model_path)
    model.eval()

    # === 4. 创建 Dataset 和 DataLoader ===
    dataset = UnlabeledDataset(args.dataset_unlabeled, tokenizer, args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, args.max_length)
    )

    # === 5. 推理 ===
    results = []
    for batch in tqdm(dataloader, desc="Generating"):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # 可设为 True + temperature 调整
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # 剥离输入部分，只保留生成内容
        input_lengths = [len(ids) for ids in input_ids]
        for i, gen_ids in enumerate(generated_ids):
            pred_ids = gen_ids[input_lengths[i]:]
            pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
            results.append({
                "original_messages": batch["original_messages"][i],
                "predicted_label": pred_text
            })

    # === 6. 保存结果 ===
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

    print(f"✅ Saved {len(results)} predictions to {args.output_file}")

if __name__ == "__main__":
    main()