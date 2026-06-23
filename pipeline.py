import os
import torch
import torch.nn.functional as F

from config import get_config 
from model import build              # Fixed based on your screenshot
from dataloader.build import build_loader  # Fixed based on your screenshot

def preprocess_pcap(pcap_file, split_dir, processed_dir):
    print(f"[*] Splitting PCAP: {pcap_file}")
    os.system(f"python3 MacspecificDP.py") # Using your Mac script!
    
    print("[*] Generating BTR flows...")
    os.system(f"python data_process/BTR_generator.py --flows_pcap_path {split_dir} --output_dir {processed_dir}")

def load_btrformer(checkpoint_path, config):
    """
    Step 2: Load the BTRFormer model and pre-trained weights.
    """
    print("[*] Loading BTRFormer Architecture...")
    model = build(config) # Using the imported 'build' function
    
    print(f"[*] Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    return model, device

def run_inference(model, data_loader, class_names, device):
    """
    Step 3: Predict!
    """
    print("[*] Running predictions...")
    predictions = []
    
    with torch.no_grad():
        for batch in data_loader:
            features = batch[0].to(device) 
            logits = model(features)
            probs = F.softmax(logits, dim=1)
            _, predicted_indices = torch.max(probs, 1)
            
            for idx in predicted_indices:
                predictions.append(class_names[idx.item()])
                
    return predictions

if __name__ == "__main__":
    # --- Configuration Variables ---
    RAW_PCAP_FILE     = "/Users/raghavdutta/Documents/ETA/Model/BTRFormer/Input/raw_file/check.pcap" # Path to the PCAP you want to analyze
    SPLIT_CAP_DIR     = "/Users/raghavdutta/Documents/ETA/Model/BTRFormer/Input/splitcap"
    PROCESSED_OUT_DIR = "/Users/raghavdutta/Documents/ETA/Model/BTRFormer/Input/processed/inference"
    CHECKPOINT_PATH   = "/Users/raghavdutta/Documents/ETA/Model/BTRFormer/output/btrformer/default/ckpt_epoch_29.pth"     # Path to HF downloaded weights
    
    # The README explicitly sets 3 classes based on config _C.MODEL.NUM_CLASSES = 3
    CLASS_NAMES = ["Normal", "Tor", "VPN"] 
    
    # 1. Preprocess the Network Traffic
    # Creates the directories if they don't exist
    os.makedirs(SPLIT_CAP_DIR, exist_ok=True)
    os.makedirs(PROCESSED_OUT_DIR, exist_ok=True)
    preprocess_pcap(RAW_PCAP_FILE, SPLIT_CAP_DIR, PROCESSED_OUT_DIR)
    
    # 2. Setup Config and DataLoader
    print("[*] Building DataLoader...")
    config = get_config() 
    _, inference_loader, _ = build_loader(config) # Kept safely in the main block!
    
    # 3. Load Model
    model, device = load_btrformer(CHECKPOINT_PATH, config)
    
    # 4. Predict
    results = run_inference(model, inference_loader, CLASS_NAMES, device)
    
    # 5. Output
    print("\n--- INFERENCE RESULTS ---")
    for i, result in enumerate(results):
        print(f"Flow {i+1}: Predicted Traffic Type -> {result}")