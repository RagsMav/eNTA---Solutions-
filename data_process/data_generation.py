# -*- coding: utf-8 -*-
"""
@File         :   data_generation.py
@Time         :   2024/10/4 18:54
@Author       :   yjn
@Contact      :   yinjn2023@zgclab.edu.cn
@Version      :   1.0
@Desc         :   Overview
@IDE          :   PyCharm
"""

import os, sys
import random
import shutil
import binascii
import scapy.all as scapy
from functools import reduce
from utils import *
from tqdm import tqdm
import multiprocessing as mp
import traceback
import math
import os
import shutil
import scapy.all as scapy
from tqdm import tqdm


def dataset_generation(pcapng_path, pcap_output_path, output_split_path):
    # pcapng_path: the path of pcapng files (if the traffic is the pacp type, pcapng_path = pcap_output_path)
    # pcap_output_path: the path of pcap files
    # output_split_path: the path of splited pcap files

    if not os.listdir(pcap_output_path):
        print("Begin to convert pcapng to pcap.")
        for _parent, _dirs, files in os.walk(pcapng_path):
            for file in files:
                if 'pcapng' in file:
                    # print(_parent + file)
                    convert_pcapng_2_pcap(_parent, file, pcap_output_path)
                else:
                    shutil.copy(_parent + "/" + file, pcap_output_path + file)

    if not os.path.exists(output_split_path):
        print("Begin to split pcap as session flows.")
        for _p, _d, files in os.walk(pcap_output_path):
            for file in files:
                split_cap(output_split_path, pcap_output_path, file)
    else:
        print(f"{output_split_path} is exist.")

def handle_raw_dataset(pcapng_or_pcap_path, output_split_path, keep_only_tls):
    clean_protocols = '"not arp and not dns and not stun and not dhcpv6 and not icmpv6 and not icmp and not dhcp and not llmnr and not nbns and not ntp and not igmp"'
    exact_five_tuple = False  # True represents each flow contains exact 5 packets

    pkts_lenxiaoyu5 = 0
    for dirpath, dirnames, filenames in os.walk(pcapng_or_pcap_path): 
        # print(dirpath, dirnames, filenames)
        for dir in dirnames:
            # print(dir)
            for dir_path, dir_name, file_names in os.walk(os.path.join(dirpath, dir)):
                # print(dir_path)
                for file in file_names:
                    print(f'current pcap file is {file}')

                    # rename pcapng as pcap
                    if file.find('.pcapng') != -1:
                        shutil.move(os.path.join(dir_path, file),
                                    os.path.join(dir_path, file.replace('.pcapng', '.pcap')))
                        file = file.replace('.pcapng', '.pcap')
                    org_file = os.path.join(dir_path, file)

                    if exact_five_tuple:
                        # truncate pcap at 100000 packets
                        new_org_file = os.path.join(dir_path, file.replace('.pcap', '_100000.pcap'))
                        os.system(f"tcpdump -r {org_file} -w {new_org_file} -c 100000")
                        os.remove(org_file)
                        org_file = new_org_file

                    # # truncate pcap at 100000 packets for self-vpn
                    # truncate_file = org_file.replace('.pcap', '.100000.pcap')
                    # if not os.path.exists(truncate_file):
                    #     os.system(f"tcpdump -r {org_file} -w {truncate_file} -c 100000")
                    # os.remove(org_file)
                    # shutil.move(truncate_file, org_file)
                    # print("After truncate org_file:", org_file)

                    # remove DHCP etc.
                    clean_file = org_file.replace('.pcap', '.clean.pcap')
                    if not os.path.exists(clean_file):
                        os.system(f"tshark -F pcap -r {org_file} -Y {clean_protocols} -w {clean_file}")
                    os.remove(org_file)
                    shutil.move(clean_file, org_file)
                    print("After clean org_file:", org_file)

                    # split by session
                    split_cap(pcap_file_path=dir_path,
                              pcap_split_path=os.path.join(output_split_path, os.path.basename(os.path.dirname(org_file))),
                              pcap_name=os.path.basename(org_file))

                    print(
    os.path.join(output_split_path, os.path.basename(os.path.dirname(org_file)), os.path.basename(org_file)))

                    if not exact_five_tuple:
                        # each session contains at least 5 packets
                        for ppp, ddd, fff in os.walk(os.path.join(output_split_path, os.path.basename(os.path.dirname(org_file)), os.path.basename(org_file))):
                            for file_ in fff:
                                second_last = os.path.basename(os.path.dirname(ppp))
                                # print(second_last)  # 输出: Cridex
                                pkts = scapy.rdpcap(os.path.join(ppp, file_), count=5)
                                if len(pkts) < 5:
                                    os.remove(os.path.join(ppp, file_))
                            break
                    else:
                        # each session contains exactly 5 packets
                        for ppp, ddd, fff in os.walk(os.path.join(output_split_path, os.path.basename(os.path.dirname(org_file)), os.path.basename(org_file))):
                            for file_ in fff:
                                pkts = scapy.rdpcap(os.path.join(ppp, file_))
                                if len(pkts) < 5:
                                    os.remove(os.path.join(ppp, file_))
                                    continue
                                flow_num = math.floor(len(pkts) / 5)
                                for i in range(flow_num):
                                    scapy.wrpcap(os.path.join(ppp, file_.replace('.pcap', f'_flow{i}.pcap')),
                                                 pkts[i * 5: i * 5 + 5])
                                os.remove(os.path.join(ppp, file_))
                            break
                break
        break


if __name__ == '__main__':
    # Point it strictly to your main data folder using an absolute path to be perfectly safe
    root_dir = '/ETA/BTRFormer/Data/'
    
    # This will automatically read from /content/raghav/BTRFormer/data/raw/
    raw_root_dir = os.path.join(root_dir, 'raw')
    
    # This will create a temporary folder to hold the split flows
    splitcap_root_dir = os.path.join(root_dir, 'splitcap')

    print(f"Starting processing from: {raw_root_dir}")
    handle_raw_dataset(pcapng_or_pcap_path=raw_root_dir, output_split_path=splitcap_root_dir, keep_only_tls=False)
