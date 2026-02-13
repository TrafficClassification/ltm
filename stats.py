import os
import argparse
from tqdm import tqdm
import argparse
import random
import os
import json
from scapy.utils import rdpcap
from scapy.all import IP, IPv6, TCP, UDP, Raw, Ether
import matplotlib.pyplot as plt
import multiprocessing as mp


def count_files_in_direct_subfolders(root_folder):

    if not os.path.exists(root_folder):
        print(f"error → {root_folder}")
        return
    if not os.path.isdir(root_folder):
        print(f"error → {root_folder}")
        return


    print(f"=== root folder:{root_folder} ===")
    print(f"{'sub folder':<30} {'file num':<10}")  
    print("-" * 40)


    for item_name in os.listdir(root_folder):
 
        item_path = os.path.join(root_folder, item_name)
        

        if os.path.isdir(item_path):

            file_count = 0
            for sub_item in os.listdir(item_path):
                sub_item_path = os.path.join(item_path, sub_item)

                if os.path.isfile(sub_item_path):
                    file_count += 1

            print(f"{item_name:<30} {file_count:<10}")


    direct_subfolders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]
    print("-" * 40)
    print(f"sub folder num: {len(direct_subfolders)}")

def process1pcap(pcap_path):
    """
    处理单个 pcap 文件，返回统计信息：
    - num_packets: int
    - [ipv4, ipv6, tcp, udp]: list of counts (each 0 or 1 for first packet)
    - file_size: int
    """
    try:
        pcap = rdpcap(pcap_path)
    except Exception as e:
        # 如果文件损坏或不是 pcap，跳过
        return None

    length_key = len(pcap)
    if length_key == 0:
        return None

    packet = pcap[0]
    proto_flags = [0, 0, 0, 0]  # [ipv4, ipv6, tcp, udp]

    if IP in packet:
        proto_flags[0] = 1
    elif IPv6 in packet:
        proto_flags[1] = 1

    if TCP in packet:
        proto_flags[2] = 1
    elif UDP in packet:
        proto_flags[3] = 1

    size_bytes = os.path.getsize(pcap_path)

    return {
        'num_packets': length_key,
        'proto_flags': proto_flags,
        'file_size': size_bytes
    }

def get_all_pcap_files(dir_folder):
    """递归获取所有 .pcap / .pcapng 文件路径"""
    pcap_files = []
    for root, _, files in os.walk(dir_folder):
        for file in files:
            if file.lower().endswith(('.pcap', '.pcapng')):
                pcap_files.append(os.path.join(root, file))
    return pcap_files

def aggregate_results(results):
    """聚合所有子进程返回的结果到 stats_on_num_pac 和 stats_on_size"""
    stats_on_num_pac = {}
    stats_on_size = {}

    for res in results:
        if res is None:
            continue
        # 聚合 num_packets 统计
        n = res['num_packets']
        if n not in stats_on_num_pac:
            stats_on_num_pac[n] = [0, 0, 0, 0]
        for i in range(4):
            stats_on_num_pac[n][i] += res['proto_flags'][i]

        # 聚合文件大小统计
        s = res['file_size']
        stats_on_size[s] = stats_on_size.get(s, 0) + 1

    return stats_on_num_pac, stats_on_size

def plot(stats_on_num_pac,stats_on_size,dir):
    
    x = list(stats_on_num_pac.keys())
    ipv4 = [v[0] for v in stats_on_num_pac.values()]
    ipv6 = [v[1] for v in stats_on_num_pac.values()]
    tcp = [v[2] for v in stats_on_num_pac.values()]
    udp = [v[3] for v in stats_on_num_pac.values()]

    # 设置图形大小
    plt.figure(figsize=(10, 6),dpi=300)

    # ===== 图1：传输层（TCP/UDP）=====
    plt.bar(x, tcp, label='TCP', color='#1f77b4')
    plt.bar(x, udp, bottom=tcp, label='UDP', color='#ff7f0e')
    #plt.yscale('log')
    plt.xlabel('Number of Packets in PCAP')
    plt.ylabel('Count of Files')
    plt.title('Transport Layer Protocols (TCP/UDP)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout() 
    plt.savefig(os.path.join(dir,"tcp_udp_distribution.png"))

    #plt.show()

    # 新建一个图
    plt.figure(figsize=(10, 6),dpi=300)

    # ===== 图2：网络层（IPv4/IPv6）=====
    plt.bar(x, ipv4, label='IPv4', color='#2ca02c')
    plt.bar(x, ipv6, bottom=ipv4, label='IPv6', color='#d62728')
    #plt.yscale('log')
    plt.xlabel('Number of Packets in PCAP')
    plt.ylabel('Count of Files')
    plt.title('Network Layer Protocols (IPv4/IPv6)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout() 
    plt.savefig(os.path.join(dir,"ipv4_ipv6_distribution.png"))

    #plt.show()
    #plotting size distribution

    x = list(stats_on_size.keys())
    y = list(stats_on_size.values())


    plt.figure(figsize=(14, 6),dpi=300)
    plt.bar(x, y, width=1.0, color='#1f77b4', edgecolor='black', linewidth=0.3)
    #plt.yscale('log')

    plt.xlabel('File Size (bytes)')
    plt.ylabel('Number of PCAP Files')
    plt.title('Distribution of PCAP File Sizes')

    # 优化 x 轴：避免太密集（可选：只显示部分 tick）
    # plt.xticks(ticks=sorted(x)[::10])  # 每10个显示一个

    # 如果 x 范围大，可以加网格便于读数
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # 自动调整布局
    plt.tight_layout()
    plt.savefig(os.path.join(dir,"size_distribution.png"))
    #plt.show()


def main():
    dir_path = "/home/xjtu/workspace/dataset/ustc_tfc2016/Malware_balanced_6000"
    
    # 获取所有 pcap 文件
    pcap_files = get_all_pcap_files(dir_path)
    print(f"Found {len(pcap_files)} pcap files.")

    if not pcap_files:
        print("No pcap files found!")
        return

    # 多进程处理
    num_workers = min(16, os.cpu_count() or 4)  # 最多16进程
    with mp.Pool(processes=num_workers) as pool:
        # 使用 tqdm 显示进度条
        results = list(tqdm(
            pool.imap(process1pcap, pcap_files),
            total=len(pcap_files),
            desc="Processing PCAPs"
        ))

    # 聚合结果
    stats_on_num_pac, stats_on_size = aggregate_results(results)

    # 排序
    stats_on_num_pac = dict(sorted(stats_on_num_pac.items()))
    stats_on_size = dict(sorted(stats_on_size.items()))
    with open(os.path.join(dir_path,"stats_on_num_pac.json"), 'w') as f:
        json.dump(stats_on_num_pac, f, indent=2)
    with open(os.path.join(dir_path,"stats_on_size.json"), 'w') as f:
        json.dump(stats_on_size, f, indent=2)
    #print("\n=== stats_on_num_pac ===")
    #print(stats_on_num_pac)
    #print("\n=== stats_on_size ===")
    #print(stats_on_size)
    # plotting
    plot(stats_on_num_pac,stats_on_size,dir_path)




if __name__ == "__main__":
    main()