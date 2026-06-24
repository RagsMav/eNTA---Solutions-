#!/usr/bin/env python3
"""
pcap_inference.py — Production inference pipeline for BTRFormer.
 
Input:  PCAP or PCAPng file
Output: CSV with per-flow predictions
 
Columns
-------
  timestamp        first-packet timestamp of the flow (UTC, ISO-8601)
  source_ip        canonical source IP (as seen by Scapy)
  destination_ip   canonical destination IP
  source_port      source TCP/UDP port
  destination_port destination TCP/UDP port
  isTor            softmax confidence for class Tor    (index 1)
  isVPN            softmax confidence for class VPN    (index 2)
  isNormal         softmax confidence for class Normal (index 0)
 
Class mapping (alphabetical — matches ImageFolder used during training)
  index 0  Normal
  index 1  Tor
  index 2  VPN
 
Pipeline (Cross-Platform Version)
--------
  1. Split into bidirectional sessions: Pure Python via Scapy (Replaces SplitCap.exe & tcpdump)
  2. Collect per-flow metadata: first-packet timestamp + 5-tuple from filename.
  3. Generate 32×32 BTR images: BTR_generator_multiprocessing
       header_len=64  payload_len=192  packet_num=4  chunk_is=False
  4. Run inference: Grayscale → ToTensor → Normalize(mean=0.5, std=0.5) → SwinTransformer
  5. Merge metadata + softmax scores → CSV.
"""
 
from __future__ import annotations
 
import argparse
import csv
import logging
import os
import shutil
import socket
import sys
import tempfile
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple
 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from scapy.utils import PcapReader, wrpcap
from scapy.layers.inet import IP, TCP, UDP
 
# ---------------------------------------------------------------------------
# Ensure repo root and data_process are importable
# ---------------------------------------------------------------------------
_ROOT      = Path(__file__).resolve().parent
_DATA_PROC = _ROOT / "data_process"
 
for _p in (str(_DATA_PROC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
 
from model.swin_transformer import SwinTransformer          # noqa: E402
from BTR_generator import BTR_generator_multiprocessing     # noqa: E402
 
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASSES: List[str] = ["Normal", "Tor", "VPN"]
_CKPT_DEFAULT = _ROOT / "output" / "btrformer" / "default" / "ckpt_epoch_29.pth"
_CLASS_DIR = "input"
_HEADER_LEN  = 64    
_PAYLOAD_LEN = 192   
_PACKET_NUM  = 4     
 
_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),                                  
    transforms.Normalize(mean=0.5, std=0.5),                
])
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pcap_btrformer")
 
@dataclass
class FlowMeta:
    timestamp: float   
    src_ip:    str
    dst_ip:    str
    src_port:  int
    dst_port:  int
    protocol:  str
 
# ---------------------------------------------------------------------------
# Step 1 — Pure Python Bidirectional Session Splitting (Replaces SplitCap & tcpdump)
# ---------------------------------------------------------------------------
 
def split_sessions_scapy(pcap: Path, flows_dir: Path) -> int:
    """
    Reads PCAP/PCAPng and extracts bidirectional flows to tiny PCAP files.
    Perfectly mimics SplitCap's '-s session -p 900' output logic.
    """
    session_dir = flows_dir / _CLASS_DIR / pcap.name
    session_dir.mkdir(parents=True, exist_ok=True)
 
    flows = {}
    log.info(f"Reading {pcap.name} with Scapy to extract sessions...")
    
    with PcapReader(str(pcap)) as reader:
        for pkt in reader:
            if IP in pkt and (TCP in pkt or UDP in pkt):
                src = str(pkt[IP].src).replace('.', '-')
                dst = str(pkt[IP].dst).replace('.', '-')
                sport = pkt.sport
                dport = pkt.dport
                proto = "TCP" if TCP in pkt else "UDP"
                
                # Sort endpoints to ensure Client->Server and Server->Client map to the same file
                endpoints = sorted([(src, sport), (dst, dport)])
                name = f"{proto}_{endpoints[0][0]}_{endpoints[0][1]}_{endpoints[1][0]}_{endpoints[1][1]}"
                
                if name not in flows:
                    flows[name] = []
                
                # Cap at 900 packets to match SplitCap's '-p 900' flag
                if len(flows[name]) < 900:
                    flows[name].append(pkt)
 
    log.info(f"Writing {len(flows)} session PCAP(s) to disk...")
    for name, pkts in flows.items():
        out_path = session_dir / f"{name}.pcap"
        wrpcap(str(out_path), pkts)
        
    return len(flows)
 
# ---------------------------------------------------------------------------
# Step 2 — Per-flow metadata collection
# ---------------------------------------------------------------------------
 
def _parse_splitcap_stem(stem: str) -> Optional[Tuple[str, str, str, int, int]]:
    parts = stem.split("_")
    if len(parts) not in (5, 6):
        return None
    proto, src_dashes, sport_s, dst_dashes, dport_s = parts[:5]  # <-- Capture 'proto' here
    try:
        src_ip = src_dashes.replace("-", ".")
        dst_ip = dst_dashes.replace("-", ".")
        socket.inet_aton(src_ip) 
        socket.inet_aton(dst_ip)
        return proto, src_ip, dst_ip, int(sport_s), int(dport_s) # <-- Return it here
    except (ValueError, OSError):
        return None
 
def collect_flow_metadata(flows_dir: Path) -> Dict[str, FlowMeta]:
    pcap_files = sorted(flows_dir.rglob("*.pcap"))
    log.info(f"Collecting flow metadata from {len(pcap_files)} session PCAP(s) ...")
 
    meta: Dict[str, FlowMeta] = {}
    skipped = 0
 
    for session_pcap in pcap_files:
        stem   = session_pcap.stem
        parsed = _parse_splitcap_stem(stem)
        if parsed is None:
            skipped += 1
            continue
        proto, src_ip, dst_ip, src_port, dst_port = parsed

        ts = 0.0
        try:
            with PcapReader(str(session_pcap)) as reader:
                first_pkt = next(iter(reader))
            ts = float(first_pkt.time)
        except StopIteration:
            pass
        except Exception as exc:
            pass
 
        meta[stem] = FlowMeta(
            timestamp=ts,
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            protocol=proto,
        )      # <-- Pass it into the dataclass here
 
    log.info(f"Metadata ready for {len(meta)} flow(s).")
    return meta
 
# ---------------------------------------------------------------------------
# Step 3 — BTR image generation
# ---------------------------------------------------------------------------
 
def generate_btr_images(flows_dir: Path, images_dir: Path) -> int:
    btr_args = types.SimpleNamespace(
        output_dir=str(images_dir),
        flows_pcap_path=str(flows_dir),
        chunk_is=False,
        sub_mat=False,
        num_processes=None,       
        mode="swin",
        header_len=_HEADER_LEN,
        payload_len=_PAYLOAD_LEN,
        packet_num=_PACKET_NUM,
    )
 
    log.info("Generating BTR images...")
    BTR_generator_multiprocessing(btr_args)
 
    count = sum(1 for _ in images_dir.rglob("*.png"))
    log.info(f"Generated {count} image(s).")
    return count
 
# ---------------------------------------------------------------------------
# Step 4 — Model loading and inference
# ---------------------------------------------------------------------------
 
def load_model(ckpt_path: Path, device: torch.device) -> SwinTransformer:
    model = SwinTransformer(
        img_size=32,
        patch_size=2,
        in_chans=1,
        num_classes=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=2,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        patch_norm=True,
        use_checkpoint=False,
        fused_window_process=False,
    )
 
    log.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
 
    state = ckpt["model"]
    rpe_keys = [k for k in state if "rpe_mlp" in k]
    for k in rpe_keys:
        state[k.replace("rpe_mlp", "cpb_mlp")] = state.pop(k)
 
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model
 
@torch.no_grad()
def classify_images(
    images_dir: Path,
    model: SwinTransformer,
    device: torch.device,
    batch_size: int = 256,
) -> List[Tuple[Path, np.ndarray]]:
    
    png_files = sorted(images_dir.rglob("*.png"))
    if not png_files:
        return []
 
    log.info(f"Running inference on {len(png_files)} image(s)...")
    results:   List[Tuple[Path, np.ndarray]] = []
    buf_imgs:  List[torch.Tensor] = []
    buf_paths: List[Path] = []
 
    for i, png in enumerate(png_files):
        img = Image.open(png)
        buf_imgs.append(_TRANSFORM(img))
        buf_paths.append(png)
 
        flush = (len(buf_imgs) == batch_size) or (i == len(png_files) - 1)
        if flush:
            batch  = torch.stack(buf_imgs).to(device)
            logits = model(batch)
            probs  = F.softmax(logits, dim=-1).cpu().numpy()  
            for path, prob in zip(buf_paths, probs):
                results.append((path, prob.astype(np.float32)))
            buf_imgs, buf_paths = [], []
 
    return results
 
# ---------------------------------------------------------------------------
# Step 5 — Merge predictions + metadata → CSV
# ---------------------------------------------------------------------------
 
def _stem_to_session_key(png_stem: str) -> str:
    prefix = _CLASS_DIR + "_"
    if png_stem.startswith(prefix):
        return png_stem[len(prefix):]
    return png_stem  
 
def _format_ts(epoch_secs: float) -> str:
    dt = datetime.fromtimestamp(epoch_secs, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")
 
def write_results(
    predictions: List[Tuple[Path, np.ndarray]],
    metadata:    Dict[str, FlowMeta],
    output_path: Path,
) -> int:
    
    rows_written = 0
    with output_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        # 1. Add "protocol" to the header row:
        writer.writerow([
            "timestamp", "protocol", "source_ip", "destination_ip",
            "source_port", "destination_port",
            "isTor", "isVPN", "isNormal",
        ])
 
        for png_path, probs in predictions:
            key  = _stem_to_session_key(png_path.stem)
            flow = metadata.get(key)
            if flow is None:
                continue
 
            # 2. Add flow.protocol to the data row:
            writer.writerow([
                _format_ts(flow.timestamp),
                flow.protocol,
                flow.src_ip,
                flow.dst_ip,
                flow.src_port,
                flow.dst_port,
                f"{probs[1]:.6f}",   # isTor
                f"{probs[2]:.6f}",   # isVPN
                f"{probs[0]:.6f}",   # isNormal
            ])
            rows_written += 1
 
    log.info(f"Wrote {rows_written} row(s) to {output_path}")
    return rows_written
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BTRFormer — cross-platform production PCAP inference pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pcap", required=True, help="Input PCAP or PCAPng file")
    p.add_argument("--output", required=True, help="Output CSV file path")
    p.add_argument("--ckpt", default=str(_CKPT_DEFAULT), help="BTRFormer checkpoint (.pth)")
    p.add_argument("--batch-size", type=int, default=256, help="Inference batch size")
    p.add_argument("--work-dir", default=None, help="Directory for intermediate files.")
    p.add_argument("--keep-work", action="store_true", help="Retain the work directory.")
    p.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    return p.parse_args()
 
def _select_device(override: Optional[str]) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
 
def main() -> None:
    args = _parse_args()
 
    pcap_path   = Path(args.pcap).resolve()
    ckpt_path   = Path(args.ckpt).resolve()
    output_path = Path(args.output).resolve()
 
    if not pcap_path.exists():
        log.error(f"PCAP not found: {pcap_path}")
        sys.exit(1)
    if not ckpt_path.exists():
        log.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)
 
    managed = args.work_dir is None
    work_dir = Path(tempfile.mkdtemp(prefix="btrformer_")) if managed else Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    flows_dir  = work_dir / "flows"
    images_dir = work_dir / "images"
    flows_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)
 
    try:
        # Step 1 — Split sessions (Pure Python)
        n_sessions = split_sessions_scapy(pcap_path, flows_dir)
        if n_sessions == 0:
            log.error("No valid TCP/UDP traffic found in PCAP.")
            sys.exit(1)
 
        # Step 2 — Metadata
        metadata = collect_flow_metadata(flows_dir)
 
        # Step 3 — Generate BTR images
        n_images = generate_btr_images(flows_dir, images_dir)
 
        # Step 4 — Classify
        device = _select_device(args.device)
        model = load_model(ckpt_path, device)
        predictions = classify_images(images_dir, model, device, batch_size=args.batch_size)
 
        # Step 5 — Write Results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_results(predictions, metadata, output_path)
 
    finally:
        if managed and not args.keep_work:
            shutil.rmtree(work_dir, ignore_errors=True)
            log.info("Work directory cleaned up.")
        else:
            log.info(f"Work directory retained: {work_dir}")
 
if __name__ == "__main__":
    main()