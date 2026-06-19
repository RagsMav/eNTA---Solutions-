# -*- coding: utf-8 -*-
"""
@File         :   utils.py
@Time         :   2024/10/4 18:54
@Author       :   yjn
@Contact      :   yinjn2023@zgclab.edu.cn
@Version      :   1.0
@Desc         :   Overview
@IDE          :   PyCharm
"""

import os, random, json, csv
import ipaddress, pickle
import re
import glob
import sys
import os
import shutil
import pyshark
from scapy.all import wrpcap, Ether
from xpinyin import Pinyin


def convert_pcapng_2_pcap(pcapng_path, pcapng_file, output_path):
    pcap_file = output_path + pcapng_file.replace('pcapng', 'pcap')
    cmd = "editcap -F pcap %s %s"
    command = cmd % (pcapng_path + pcapng_file, pcap_file)
    os.system(command)
    return 0


def split_cap(pcap_file_path, pcap_split_path, pcap_name, split_way='bidirection'):
    # pcap_split_path + pcap_label + "/" + pcap_name is output
    # pcap_file_path+pcap_name is input
    if not os.path.exists(pcap_split_path):
        os.makedirs(pcap_split_path, exist_ok=True)

    output_path = os.path.join(pcap_split_path, pcap_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    split_way = "session" if split_way == 'bidirection' else "flow"
    raw_pcap_path = os.path.join(pcap_file_path, pcap_name)
    print(f"raw_pcap_path:{raw_pcap_path}, output_path:{output_path}")
    cmd = f"mono ./SplitCap.exe -r {raw_pcap_path} -s {split_way} -o {output_path} -p {900}"
    os.system(cmd)
    return output_path


def cut(obj, sec):
    result = [obj[i:i + sec] for i in range(0, len(obj), sec)]
    try:
        remanent_count = len(result[0]) % 4
    except Exception as e:
        remanent_count = 0
        print("cut datagram error!")
    if remanent_count == 0:
        pass
    else:
        result = [obj[i:i + sec + remanent_count] for i in range(0, len(obj), sec + remanent_count)]
    return result


# generate random ipv4 address
def random_ipv4():
    IPV4_MAX = ipaddress.IPv4Address._ALL_ONES
    ip_int = random.randint(0, IPV4_MAX)
    ip_str = ipaddress.IPv4Address._string_from_ip_int(ip_int)
    return ip_str

def makedir(path):
    try:
        os.mkdir(path)
    except Exception as E:
        pass



