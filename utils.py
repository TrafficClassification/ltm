"""
*author: KangShiqi
*description: this file contains tool functions to help with the research of traffic classification 
*note the splitcap.exe and tshark(or wireshark) are required to use some of the functions
"""
from typing import Optional, Tuple, Union
import os
import subprocess
import tqdm


def pcapng2pcap(pcapng_file: str, preserve: bool) -> str:
    """
    Convert a .pcapng file to a .pcap file using tshark.
    
    :param pcapng_file: Path to the .pcapng file
    :param preserve: Whether to keep the original .pcapng file
    :return: Path to the generated .pcap file
    """
    if not pcapng_file.endswith(".pcapng"):
        raise ValueError("pcapng_file must end with .pcapng")
    if not os.path.exists(pcapng_file):
        raise FileNotFoundError(f"pcapng_file {pcapng_file} not found")
    
    pcap_file = pcapng_file.replace(".pcapng", ".pcap")
    
    # Use tshark to convert .pcapng to .pcap
    subprocess.run(["tshark", "-r", pcapng_file, "-F", "libpcap", "-w", pcap_file])
    
    if not preserve:
        os.remove(pcapng_file)
    
    return pcap_file


def flow_generation(pcap_file: str, output_dir: str, mode: str) -> str:
    """
    Generate flows from a .pcap file using SplitCap.exe via Mono.
    
    :param pcap_file: Path to the .pcap file
    :param output_dir: Directory to save generated flows
    :param mode: Flow generation mode (e.g., session, flow)
    :return: Path to the directory containing the generated flows
    """
    # 确保 SplitCap.exe 的路径正确（假设放在当前目录下）
    splitcap_exe_path = os.path.join(os.getcwd(), "SplitCap.exe")
    
    if not os.path.exists(splitcap_exe_path):
        raise FileNotFoundError(f"SplitCap.exe not found at {splitcap_exe_path}")
    
    pcap_name = os.path.splitext(os.path.basename(pcap_file))[0]
    output_path = os.path.join(output_dir, pcap_name)
    os.makedirs(output_path, exist_ok=True)
    
    # 使用 mono 运行 SplitCap.exe
    try:
        cmd = ["mono", splitcap_exe_path, "-r", pcap_file, "-o", output_path, "-s", mode]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error generating flows for {pcap_file}: {e}")
        return output_path
    
    return output_path

    
