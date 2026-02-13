import os
import glob
import multiprocessing as mp
from utils import pcapng2pcap, flow_generation
from tqdm import tqdm

data_path = "/home/xjtu/workspace/dataset/Tor"
output_path = "/home/xjtu/workspace/dataset/iscx_tor"

def convert_pcapng(pcapng_file_name):
    pcap_file_name = pcapng_file_name.replace(".pcapng", ".pcap")
    pcapng2pcap(pcapng_file_name, preserve=True)

def process_pcap(pcap_file_name):
    flow_generation(pcap_file_name, output_path, mode="session")

if __name__ == "__main__":
    pcapng_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".pcapng"):
                pcapng_files.append(os.path.join(root, file))

    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        list(tqdm(pool.imap(convert_pcapng, pcapng_files),
                  total=len(pcapng_files),
                  desc="Converting pcapng files"))


    pcap_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".pcap"):
                pcap_files.append(os.path.join(root, file))


    with mp.Pool(processes=mp.cpu_count()-4) as pool:
        list(tqdm(pool.imap(process_pcap, pcap_files),
                  total=len(pcap_files),
                  desc="Processing pcap flows"))