import json
import random

def add_noise_to_list(
    data,
    shuffle_prob=0.1,
    delete_ratio=0.1,
    insert_ratio=0.1,
    noise_pool=None
):
    data = data.copy()  # 不修改原列表
    
    # 1. 可能打乱
    if random.random() < shuffle_prob:
        random.shuffle(data)
    
    # 2. 随机删除
    n_del = int(len(data) * delete_ratio)
    if n_del > 0:
        to_del = random.sample(range(len(data)), min(n_del, len(data)))
        to_del = sorted(to_del, reverse=True)  # 从后往前删，避免索引错位
        for idx in to_del:
            data.pop(idx)
    
    # 3. 随机插入噪声
    n_ins = int(len(data) * insert_ratio)
    for _ in range(n_ins):
        if noise_pool:
            noise = random.choice(noise_pool)
        else:
            noise = random.choice(data) if data else 0  # 默认用已有元素
        pos = random.randint(0, len(data))
        data.insert(pos, noise)
    
    return data
if __name__ == "__main__":
    dataset = []
    with open('/home/xjtu/workspace/ltm/dataset/first_n/pac4/vpn_services_1token_test.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            tsp = data['messages'][1]['content'].split("<pck>")[0]
            flow_data = data['messages'][1]['content'].split("<pck>")[1:]
            new_flow_data = "<pck>"+"<pck>".join(add_noise_to_list(flow_data))
            data['messages'][1]['content'] = tsp+new_flow_data
            dataset.append(data)
    with open('/home/xjtu/workspace/ltm/dataset/ablation/ustc_benign/vpn_services_1token_mixture_test.jsonl', 'w') as f:
        for data in dataset:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        
