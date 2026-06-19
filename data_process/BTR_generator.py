import argparse
import os
import glob
import binascii
from PIL import Image
from scapy.all import *
from tqdm import tqdm
import numpy as np
from utils import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from scapy.utils import PcapReader


# +
def read_packets_with_limit(pcap_dir, max_packets):
    packets = []
    try:
        with PcapReader(pcap_dir) as pcap_reader:
            for i, pkt in enumerate(pcap_reader):
                if i >= max_packets:
                    break
                packets.append(pkt)
    except Exception as e:
        print(f"Error reading {pcap_dir}: {e}")
    return packets

def get_normalized_ip_pair(src, dst):
    # Sorts the IPs alphabetically so A->B and B->A always match
    if src < dst:
        return (src, dst)
    return (dst, src)

def get_args_parser():
    parser = argparse.ArgumentParser('BTR data gen process', add_help=False)

    # Dataset parameters
    parser.add_argument('--flows_pcap_path', default='', type=str,
                        help='flows dataset path')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./data/',
                        help='path where to tensorboard log')
    parser.add_argument('--chunk_is', default=False, help='whether to chunk, while data os Tor need to chunk')
    parser.add_argument('--sub_mat', default=False, help='whether to sub_mat, 40*4*10')
    parser.add_argument('--mode', default='swin', type=str,
                        help='mode to 40*40 or 40*4*10 or 4*40*2 + (4*4*10*2)*3 or 16*16 + 16*16 + 16*16 + 16*16')
    parser.add_argument('--header_len', default=64, type=int,
                        help='header_len')
    parser.add_argument('--payload_len', default=192, type=int,
                        help='payload_len')
    parser.add_argument('--num_processes', default=None, help='processes number')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--packet_num', default=4, type=int)

    return parser


# -

def read_BTR_bytes_without_directions(pcap_dir, header_len, payload_len, chunk_is, packet_num):
    header_len = header_len * 2
    payload_len = payload_len * 2
    packets = read_packets_with_limit(pcap_dir, max_packets=packet_num if not chunk_is else packet_num * 25)

    data = []
    for packet in packets:
        if IP in packet:
            packet[IP].src = random_ipv4()
            packet[IP].dst = random_ipv4()
        if TCP in packet:
            packet[TCP].sport = 0
            packet[TCP].dport = 0
        elif UDP in packet:
            packet[UDP].sport = 0
            packet[UDP].dport = 0

        header = (binascii.hexlify(bytes(packet['IP']))).decode()
        try:
            payload = (binascii.hexlify(bytes(packet['Raw']))).decode()
            header = header.replace(payload, '')
        except:
            payload = ''

        if len(header) > header_len:
            header = header[:header_len]
        elif len(header) < header_len:
            header += '0' * (header_len - len(header))
        if len(payload) > payload_len:
            payload = payload[:payload_len]
        elif len(payload) < payload_len:
            payload += '0' * (payload_len - len(payload))
        data.append((header, payload))

        if len(data) >= packet_num:
            break

    if len(data) < packet_num:
        for i in range(packet_num - len(data)):
            data.append(('0' * header_len, '0' * payload_len))

    final_data = ''
    for h, p in data:
        final_data += h
        final_data += p
    return final_data


def read_BTR_bytes_with_directions(pcap_dir, header_len, payload_len, chunk_is, packet_num):
    header_len = header_len * 2
    payload_len = payload_len * 2
    packets = read_packets_with_limit(pcap_dir, max_packets=packet_num if not chunk_is else packet_num * 25)

    ip_pair_mapping = {}

    data = []
    for packet in packets:
        if IP in packet:
            original_src = packet[IP].src
            original_dst = packet[IP].dst
            normalized_pair = get_normalized_ip_pair(original_src, original_dst)

            if normalized_pair not in ip_pair_mapping:
                new_src, new_dst = random_ipv4(), random_ipv4()
                if original_src != original_dst:
                    while new_src == new_dst:
                        new_dst = random_ipv4()
                ip_pair_mapping[normalized_pair] = (new_src, new_dst)

            if packet[IP].src == normalized_pair[0]:
                packet[IP].src, packet[IP].dst = ip_pair_mapping[normalized_pair]
            else:
                packet[IP].dst, packet[IP].src = ip_pair_mapping[normalized_pair]

        if TCP in packet:
            packet[TCP].sport = 0
            packet[TCP].dport = 0
        elif UDP in packet:
            packet[UDP].sport = 0
            packet[UDP].dport = 0

        if 'IP' in packet:
            header = (binascii.hexlify(bytes(packet['IP']))).decode()
        elif 'IPv6' in packet:
            header = (binascii.hexlify(bytes(packet['IPv6']))).decode()
        else:
            continue  # Skip strange packets like ARP that don't have IP layers
        try:
            payload = (binascii.hexlify(bytes(packet['Raw']))).decode()
            header = header.replace(payload, '')
        except:
            payload = ''

        if len(header) > header_len:
            header = header[:header_len]
        elif len(header) < header_len:
            header += '0' * (header_len - len(header))
        if len(payload) > payload_len:
            payload = payload[:payload_len]
        elif len(payload) < payload_len:
            payload += '0' * (payload_len - len(payload))
        data.append((header, payload))

        if not chunk_is:
            if len(data) >= packet_num:
                break
        else:
            if len(data) >= packet_num * 25:
                break

    if len(data) < packet_num:
        for i in range(packet_num - len(data)):
            data.append(('0' * header_len, '0' * payload_len))

    final_data = ''
    for h, p in data:
        final_data += h
        final_data += p
    return final_data


def BTR_generator(flows_pcap_path, output_path, chunk_is):
# 1. Fix the path (use exactly what we pass in the terminal)
    splitcap_flows_pcap_path = flows_pcap_path
    print(f"Looking in: {splitcap_flows_pcap_path}")

    # 2. Fix the glob depth (look inside the class folders like /VPN/)
    flows = glob.glob(os.path.join(splitcap_flows_pcap_path, "*/*.pcap"))
    print(f"Found {len(flows)} pcap files to process!")

    makedir(output_path)
    makedir(output_path + "/train")
    makedir(output_path + "/test")
    
    classes = glob.glob(os.path.join(splitcap_flows_pcap_path, "*"))
    
    # 3. Nuke get_folder_names and extract paths safely
    for cla in tqdm(classes):
        if os.path.isdir(cla):
            makedir(os.path.join(output_path, os.path.basename(cla)))

    for flow in tqdm(flows):
        content = read_BTR_bytes(flow)
        content = np.array([int(content[i:i + 2], 16) for i in range(0, len(content), 2)])
        fh = np.reshape(content, (400, 4))
        
        # Get the 'VPN'/'Tor' name and actual file name
        class_name = os.path.basename(os.path.dirname(flow))
        base_file = os.path.basename(flow)
        
        file_name = f"{class_name}_{base_file.replace('.pcap', '.png')}"
        save_image_from_array(fh, output_path, folder_name=class_name, file_name=file_name)


def save_image_from_array(content, output_path, folder_name, file_name):
    fh = np.uint8(content)
    im = Image.fromarray(fh)

    save_path = os.path.join(output_path, folder_name, file_name)
    im.save(save_path)


def assemble_matrices(matrixs):
    packet_num = len(matrixs)
    if packet_num == 0:
        return None

    n = int(round(np.sqrt(packet_num)))
    if n * n != packet_num:
        raise ValueError("Number of matrices must be a perfect square")

    small_height, small_width = matrixs[0].shape

    final_matrix = np.zeros((n * small_height, n * small_width), dtype=matrixs[0].dtype)

    for i in range(n):
        row = np.hstack(matrixs[i * n: (i + 1) * n])
        final_matrix[i * small_height: (i + 1) * small_height, :] = row

    return final_matrix


def reshape_content(content, packet_len, packet_num, mode='default'):
    packets = []
    for i in range(packet_num):
        start_index = i * packet_len
        end_index = (i + 1) * packet_len
        packets.append(content[start_index:end_index])

    matrixs = []
    # Process each packet
    for i in range(packet_num):
        matrixs.append(process_packet(packets[i], packet_len, packet_num))

    final_matrix = assemble_matrices(matrixs)

    return final_matrix


def process_packet(packet, packet_len, packet_num):
    assert len(packet) == packet_len, "Each packet must be 256 bytes."

    header = packet[:64]  #
    payload = packet[64:]

    header_matrix = header.reshape(4, 4, 4)
    final_header = np.hstack([header_matrix[i, :, :] for i in range(4)])

    payload_matrix = payload.reshape(12, 4, 4)
    final_payload = np.vstack([np.hstack(payload_matrix[i:i + 4]) for i in range(0, len(payload_matrix), 4)])

    result_matrix = np.vstack([final_header, final_payload])  # Stack them vertically
    return result_matrix


def process_flow_chunk(args):
    flow, output_path, chunk_is, sub_mat, mode, header_len, payload_len, packet_num = args
    packet_len = header_len + payload_len
    try:
        content = read_BTR_bytes_with_directions(flow, header_len=header_len, payload_len=payload_len, chunk_is=chunk_is, packet_num=packet_num)
        content = np.array([int(content[i:i + 2], 16) for i in range(0, len(content), 2)])

        # Safely extract class ('VPN') and filename without custom functions
        class_name = os.path.basename(os.path.dirname(os.path.dirname(flow)))
        base_file = os.path.basename(flow)

        if not chunk_is:
            fh = reshape_content(content, packet_len, packet_num, mode)
            save_image_from_array(fh, output_path, folder_name=class_name,
                                  file_name=f"{class_name}_{base_file.replace('.pcap', '.png')}")
        else:
            total_bytes = len(content)
            num_chunks = total_bytes // (packet_len * packet_num)

            for chunk_idx in range(num_chunks):
                start = chunk_idx * (packet_len * packet_num)
                end = start + (packet_len * packet_num)
                chunk_content = content[start:end]

                fh = reshape_content(chunk_content, packet_len, packet_num, mode)
                save_image_from_array(fh, output_path, folder_name=class_name,
                                      file_name=f"{class_name}_{base_file.replace('.pcap', f'_{chunk_idx}.png')}")    
    except Exception as e:
        print(f"Error processing {flow}: {e}")


def BTR_generator_multiprocessing(args):
    output_path = args.output_dir
    flows_pcap_path = args.flows_pcap_path
    chunk_is = args.chunk_is
    sub_mat = args.sub_mat
    num_processes = args.num_processes
    mode = args.mode
    header_len = args.header_len
    payload_len = args.payload_len
    packet_num = args.packet_num

    # 1. Use the exact path passed from the terminal
    splitcap_flows_pcap_path = flows_pcap_path
    
    # 2. Fix the glob depth (look exactly 1 subfolder deep for pcaps)
    flows = glob.glob(os.path.join(splitcap_flows_pcap_path, "*/*/*.pcap"))
    print(f"SUCCESS: Found {len(flows)} pcap files to process!")

    makedir(output_path)
    makedir(os.path.join(output_path, "train"))
    makedir(os.path.join(output_path, "test"))

    # 3. Safely create class output folders (VPN, Normal, Tor)
    classes = glob.glob(os.path.join(splitcap_flows_pcap_path, "*"))
    for cla in classes:
        if os.path.isdir(cla):
            makedir(os.path.join(output_path, os.path.basename(cla)))    
    
    if num_processes is None:
        num_processes = cpu_count()

    import psutil
    if psutil.virtual_memory().available < 2 * 1024 ** 3:
        num_processes = max(num_processes // 2, 1)

    with Pool(processes=int(num_processes)) as pool:
        task_args = [(flow, output_path, chunk_is, sub_mat, mode, header_len, payload_len, packet_num) for flow in
                     flows]

        for _ in tqdm(pool.imap_unordered(process_flow_chunk, task_args), total=len(flows)):
            pass


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    print(args)
    BTR_generator_multiprocessing(args)
