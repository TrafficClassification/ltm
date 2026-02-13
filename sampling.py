import json

input_file = '/home/xjtu/workspace/ltm/dataset/first_n/pac4/ustc_benign_1token_test.jsonl'   # 替换为你的输入文件路径
output_file = '/home/xjtu/workspace/ltm/dataset/n_only/ustc_benign_1token_tcp.jsonl'  # 输出文件路径
n=8
with open(input_file, 'r', encoding='utf-8') as fin, \
     open(output_file, 'w', encoding='utf-8') as fout:

    for line in fin:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            user_content = data.get("messages", [{}])[1].get("content", "")
            if f"tcp" in user_content:
                fout.write(line + '\n')
        except (json.JSONDecodeError, IndexError, KeyError):
            continue  # 跳过格式错误的行