import json
import argparse
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file', type=str, required=True, help='Path to prediction.jsonl')
    parser.add_argument('--show_mismatches', type=int, default=5, help='Number of mismatched examples to show')
    return parser.parse_args()

def main():
    args = parse_args()
    
    total = 0
    exact_match = 0
    mismatches = []
    predict = []
    real = []

    with open(args.prediction_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Evaluating"):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                
                # 提取 ground truth assistant content
                ground_truth = ""
                for msg in item.get("original_messages", []):
                    
                    if msg["role"] == "assistant":
                        ground_truth = msg["content"].strip()
                        break
                
                predicted = item.get("predicted_label", "").strip()#.rstrip(".,!?:;，。！？；：")
                total += 1
                predict.append(predicted)
                real.append(ground_truth)

                if predicted == ground_truth:
                    exact_match += 1
                else:
                    if len(mismatches) < args.show_mismatches:
                        mismatches.append({
                            "ground_truth": ground_truth,
                            "predicted": predicted,
                            "messages": item["original_messages"]
                        })
            except Exception as e:
                print(f"Error processing line: {line} | {e}")

    # 输出结果
    em_rate = exact_match / total if total > 0 else 0
    print("\n" + "="*80)
    print(f"✅ Total samples: {total}")
    print(f"🎯 Exact Match (EM): {exact_match} / {total} = {em_rate:.2%}")
    #print(f"🎯 Accuracy: {accuracy_score(real, predict):.2%}")
    #print(f"🎯 Precision: {precision_score(real, predict, average='weighted'):.2%}")
    #print(f"🎯 Recall: {recall_score(real, predict, average='weighted'):.2%}")
    #print(f"🎯 F1 Score: {f1_score(real, predict, average='weighted'):.2%}")
    report = classification_report(real, predict,output_dict=True)
    #df = pd.DataFrame(report).transpose()
    #df.to_csv(args.prediction_file.replace(".jsonl", "_report.csv"), index= True)
    with open(args.prediction_file.replace(".jsonl", "_report.json"), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    
    print("="*80)

    if mismatches:
        print(f"\n🔍 Top {len(mismatches)} Mismatched Examples:")
        for i, ex in enumerate(mismatches, 1):
            print(f"\n--- Example {i} ---")
            print(f"Ground Truth: {repr(ex['ground_truth'])}")
            print(f"Prediction   : {repr(ex['predicted'])}")
            # 可选：打印上下文
            # print("Context:", [m for m in ex['messages'] if m['role'] != 'assistant'])
        with open(args.prediction_file.replace(".jsonl", "_mismatches.jsonl"), 'w', encoding='utf-8') as f:
            for ex in mismatches:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    else:
        print("\n🎉 All predictions matched ground truth!")

if __name__ == "__main__":
    main()