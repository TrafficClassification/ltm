from tqdm import tqdm
import argparse
import random
import os
import json
from scapy.utils import rdpcap
from scapy.all import IP, IPv6, TCP, UDP, Raw, Ether
import binascii
import multiprocessing as mp
from transformers import AutoTokenizer

MODEL_NAME = "/home/xjtu/workspace/ltm/models/Qwen3_4b"
MAX_PACKET_NUMBER = 4
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def one_tokenize(labels):
    one_token_label = {}
    for label in labels:
        one_token_label[label] = tokenizer.decode(tokenizer(label)["input_ids"][-1])
    return one_token_label

def num_labeling(labels):
    num_labels = {}
    for i,label in enumerate(labels):
        num_labels[label] = str(i)
    
    return num_labels


    


def write_dataset(dataset, output_path):
    with open(output_path, "w", encoding="utf-8") as fin:
        for data in dataset:
            json.dump(data, fin)
            fin.write("\n")

def save_dataset(args, train_dataset, valid_dataset, test_dataset):
    write_dataset(train_dataset, os.path.join(args.output_path, args.output_name + "_train.jsonl"))
    write_dataset(valid_dataset, os.path.join(args.output_path, args.output_name + "_valid.jsonl"))
    write_dataset(test_dataset, os.path.join(args.output_path, args.output_name + "_test.jsonl"))

def write_labels(labels, output_path):
    label_dict = {}
    for i, label in enumerate(labels):
        label_dict[label] = i
    with open(output_path, "w", encoding="utf-8") as fin:
        json.dump(label_dict, fin, indent=4, separators=(',', ': '))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="raw dataset path", required=True)
    parser.add_argument("--output_path", type=str, help="output dataset path", required=True)
    parser.add_argument("--output_name", type=str, help="output dataset name", required=True)
    parser.add_argument("--system_prompt", type=str, help="system prompt", required=True)
    parser.add_argument("--num_workers", type=int, default=4, help="number of multiprocessing workers")
    args = parser.parse_args()
    return args

def extract_ip_transport_header(packet):
    if IP in packet:
        ip_layer = packet[IP]
        ip_header = ip_layer.build()[:ip_layer.ihl * 4]
        if TCP in packet:
            tcp_layer = packet[TCP]
            tcp_header = tcp_layer.build()[:tcp_layer.dataofs * 4]
            return "ip&tcp", ip_header, tcp_header
        elif UDP in packet:
            udp_layer = packet[UDP]
            udp_header = udp_layer.build()[:8]
            return "ip&udp", ip_header, udp_header
    elif IPv6 in packet:
        ipv6_header = packet[IPv6].build()[:40]
        if TCP in packet:
            tcp_layer = packet[TCP]
            tcp_header = tcp_layer.build()[:tcp_layer.dataofs * 4]
            return "ipv6&tcp", ipv6_header, tcp_header
        elif UDP in packet:
            udp_layer = packet[UDP]
            udp_header = udp_layer.build()[:8]
            return "ipv6&udp", ipv6_header, udp_header
    return None
def extract_ip_transport_header_ip_masked(packet):
    if IP in packet:
        ip_layer = packet[IP]
        ip_header = ip_layer.build()[:ip_layer.ihl * 4]
        
        # 将IPv4头部中的源IP和目的IP替换为0
        ip_header_bytes = bytearray(ip_header)
        # 源IP地址位置：12-15字节
        for i in range(12, 16):
            if i < len(ip_header_bytes):
                ip_header_bytes[i] = 0
        # 目的IP地址位置：16-19字节  
        for i in range(16, 20):
            if i < len(ip_header_bytes):
                ip_header_bytes[i] = 0
        ip_header = bytes(ip_header_bytes)
        
        if TCP in packet:
            tcp_layer = packet[TCP]
            tcp_header = tcp_layer.build()[:tcp_layer.dataofs * 4]
            return "ip&tcp", ip_header, tcp_header
        elif UDP in packet:
            udp_layer = packet[UDP]
            udp_header = udp_layer.build()[:8]
            return "ip&udp", ip_header, udp_header
            
    elif IPv6 in packet:
        ipv6_layer = packet[IPv6]
        ipv6_header = ipv6_layer.build()[:40]
        
        # 将IPv6头部中的源IP和目的IP替换为0
        ipv6_header_bytes = bytearray(ipv6_header)
        # IPv6源地址位置：8-23字节（128位）
        for i in range(8, 24):
            if i < len(ipv6_header_bytes):
                ipv6_header_bytes[i] = 0
        # IPv6目的地址位置：24-39字节（128位）
        for i in range(24, 40):
            if i < len(ipv6_header_bytes):
                ipv6_header_bytes[i] = 0
        ipv6_header = bytes(ipv6_header_bytes)
        
        if TCP in packet:
            tcp_layer = packet[TCP]
            tcp_header = tcp_layer.build()[:tcp_layer.dataofs * 4]
            return "ipv6&tcp", ipv6_header, tcp_header
        elif UDP in packet:
            udp_layer = packet[UDP]
            udp_header = udp_layer.build()[:8]
            return "ipv6&udp", ipv6_header, udp_header
            
    return None

def process1pcap(pcap_path, sys_prompt, label):
    packet_list = []
    cnt = 0
    packets = rdpcap(pcap_path)
    
    total_packets = len(packets)
    #if total_packets != 10:
    #    return None
    time_range = packets[-1].time - packets[0].time

    for i, packet in enumerate(packets):
        if i >= MAX_PACKET_NUMBER:
            break
        ip_transport_type, ip_header, header = extract_ip_transport_header(packet)
        if ip_transport_type is None:
            continue
        #if "udp" not in ip_transport_type:
        #    return None
        cnt += 1
        packet_list.append(
            ip_transport_type + "<Internet>" +
            binascii.hexlify(ip_header).decode() + 
            "<Transport>" +
            binascii.hexlify(header).decode()
        )
        #packet_list.append(
        #    binascii.hexlify(ip_header).decode()+
        #    binascii.hexlify(header).decode()
        #)
    packet_headers = "<pck>" + "<pck>".join(packet_list)
    #packet_headers = " ".join(packet_list)
    #packet_headers = "<stats>" + f"First {cnt} packets of total {total_packets} packets. Time span: {time_range:.2f} seconds. " + "<stats>" + packet_headers

    #packet_headers = f"First {cnt} packets of total {total_packets} packets. Time span: {time_range:.2f} seconds." + packet_headers
    return {
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": packet_headers},
            {"role": "assistant", "content": label}
        ]
    }

# 用于多进程的包装函数（必须在全局作用域）
def _process_wrapper(args_tuple):
    pcap_path, sys_prompt, label = args_tuple
    return process1pcap(pcap_path, sys_prompt, label)

def traffic_classification_preprocess(args):
    dataset = []
    labels = []
    tasks = []

    files = os.listdir(args.input)
    labels.extend(files)
    write_labels(labels, os.path.join(args.output_path, args.output_name + "_label.json"))
    # 测试 1 token 标签
    one_token_labels = one_tokenize(labels)
    num_labels = num_labeling(labels)
    category_1token = ", ".join(one_token_labels.values())
    category_num = ", ".join(num_labels.values())
    category = ", ".join(labels)
    onetoken_PROMPT = f"Classify this network traffic data into one application category: {category_1token}. Output the category name ONLY. Do NOT add any punctuation, explanation, or extra text."
    num_prompt = f"Classify this network traffic data into one application category: {category_num}. Output the category name ONLY. Do NOT add any punctuation, explanation, or extra text."
    PROMPT = f"Classify this network traffic data into one application category: {category}. Output the category name ONLY. Do NOT add any punctuation, explanation, or extra text."
    
    # 收集所有任务
    for root, dirs, files in os.walk(args.input):
        for file in files:
            pcap_path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(pcap_path))
            tasks.append((pcap_path, onetoken_PROMPT,one_token_labels[label]))


    # 多进程处理
    with mp.Pool(processes=args.num_workers) as pool:
        results = list(tqdm(
            pool.imap(_process_wrapper, tasks),
            total=len(tasks),
            desc="Processing pcap files"
        ))

    dataset = [item for item in results if item is not None]
    random.shuffle(dataset)

    # 划分数据集
    n = len(dataset)
    train_end = int(n * 0.8)
    valid_end = int(n * 0.9)
    train_dataset = dataset[:train_end]
    valid_dataset = dataset[train_end:valid_end]
    test_dataset = dataset[valid_end:]

    save_dataset(args, train_dataset, valid_dataset, test_dataset)
    #write_dataset(dataset, os.path.join(args.output_path, args.output_name + "_fullset.jsonl"))

if __name__ == "__main__":
    args = get_args()
    traffic_classification_preprocess(args)

