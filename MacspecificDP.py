import os
import numpy as np
from PIL import Image
from scapy.all import PcapReader, IP, TCP, UDP

def process_pcap_to_png(input_pcap, output_dir, max_images=1000):
    """Reads a raw PCAP, extracts TCP/UDP payloads, and saves 16x16 PNGs."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing: {input_pcap}")
    
    count = 0
    # Stream the packets one by one so we don't crash your Mac's RAM
    with PcapReader(input_pcap) as pcap_iter:
        for packet in pcap_iter:
            if count >= max_images:
                break
                
            # We only care about packets with actual data payloads
            if packet.haslayer(IP) and (packet.haslayer(TCP) or packet.haslayer(UDP)):
                raw_bytes = bytes(packet[IP])
                
                # BTRFormer requires exactly 256 bytes for a 16x16 matrix
                if len(raw_bytes) < 256:
                    raw_bytes = raw_bytes.ljust(256, b'\x00') # Pad short packets
                else:
                    raw_bytes = raw_bytes[:256]               # Trim long packets
                
                # Convert bytes to an image matrix
                img_data = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((16, 16))
                img = Image.fromarray(img_data, 'L')

                img = img.resize((32, 32), Image.NEAREST)
                
                # Save the image
                img.save(os.path.join(output_dir, f"flow_{count}.png"))
                count += 1
                
                if count % 200 == 0:
                    print(f"  Generated {count} images...")

    print(f"Done! Saved {count} images to {output_dir}\n")

if __name__ == "__main__":
    # Base directories
    base_raw = "/Model/BTRFormer/Dat/raw"
    base_processed = "/Model/BTRFormer/Dat/processed/train"

    # Map the specific PCAP files you downloaded to their output folders
    # Update "Normal.pcap" etc. to the exact names of the files you put in Dat/raw/
    
    categories = {
        "Normal": "profile_3_Normal.pcap", 
        "Tor": "profile_1_Tor.pcap",
        "VPN": "profile_2_VPN.pcap"
    }

    for category, filename in categories.items():
        input_path = os.path.join(base_raw, category, filename)
        output_path = os.path.join(base_processed, category)
        
        if os.path.exists(input_path):
            process_pcap_to_png(input_path, output_path, max_images=1000)
        else:
            print(f"Could not find {input_path} - skipping.")
