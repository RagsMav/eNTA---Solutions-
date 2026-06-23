import os
import torch
import torch.nn.functional as F

# Importing modules based on the repository's structure
# Note: You may need to adjust import names slightly depending on the exact classes 
# defined inside models_BTR.py and config.py
from config import get_config # Assuming config.py has a config builder
from models_BTR import build_model # Assuming there is a build_model method or BTRFormer class
from dataloader import build_dataloader # Assuming a dataloader script is present

def preprocess_pcap(pcap_file, split_dir, processed_dir):
    """
    Step 1: Convert raw PCAP to processed BTR features using the repo's existing scripts.
    """
    print(f"[*] Splitting PCAP: {pcap_file}")
    # 1. Run the PCAP splitter (Ensure mono is installed as per the README)
    # Note: data_generation.py handles the splitting logic
    os.system(f"python3 data_process/data_generation.py --input {pcap_file}")
    
    print("[*] Generating BTR flows...")
    # 2. Run the BTR generator to convert splits to the model's expected format
    os.system(f"python data_process/BTR_generator.py --flows_pcap_path {split_dir} --output_dir {processed_dir}")

def load_btrformer(checkpoint_path, config):
    """
    Step 2: Load the BTRFormer model and pre-trained weights.
    """
    print("[*] Loading BTRFormer Architecture...")
    # Initialize the model using the repository's configuration
    model = build_model(config) 
    
    print(f"[*] Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Depending on how it was saved, you might need checkpoint['model'] or just checkpoint
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    
    # Move model to GPU if available and set to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    return model, device

def run_inference(model, data_loader, class_names, device):
    """
    Step 3: Run the processed data through the model to get class predictions.
    """
    print("[*] Running predictions...")
    predictions = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Assuming the dataloader returns (features, labels/dummy_labels)
            features = batch[0].to(device) 
            
            # Forward pass
            logits = model(features)
            
            # Apply softmax to get confidence probabilities
            probs = F.softmax(logits, dim=1)
            
            # Get the index of the highest probability
            _, predicted_indices = torch.max(probs, 1)
            
            for idx in predicted_indices:
                predictions.append(class_names[idx.item()])
                
    return predictions

if __name__ == "__main__":
    # --- Configuration Variables ---
    RAW_PCAP_FILE     = "./Data/raw/sample_traffic.pcap" # Path to the PCAP you want to analyze
    SPLIT_CAP_DIR     = "./Data/splitcap/"
    PROCESSED_OUT_DIR = "./Data/processed/inference/"
    CHECKPOINT_PATH   = "./btrformer_checkpoint.pth"     # Path to HF downloaded weights
    
    # The README explicitly sets 3 classes based on config _C.MODEL.NUM_CLASSES = 3
    CLASS_NAMES = ["Normal", "Tor", "VPN"] 
    
    # 1. Preprocess the Network Traffic
    # Creates the directories if they don't exist
    os.makedirs(SPLIT_CAP_DIR, exist_ok=True)
    os.makedirs(PROCESSED_OUT_DIR, exist_ok=True)
    
    preprocess_pcap(RAW_PCAP_FILE, SPLIT_CAP_DIR, PROCESSED_OUT_DIR)
    
    # 2. Setup Config and DataLoader
    # Load configuration parameters (e.g., image size, patch size, etc.)
    config = get_config() 
    
    # Build a dataloader pointing to the newly processed directory
    # (is_train=False ensures no random augmentations are applied)
    inference_loader = build_dataloader(is_train=False, config=config, data_path=PROCESSED_OUT_DIR)
    
    # 3. Load the Model
    model, device = load_btrformer(CHECKPOINT_PATH, config)
    
    # 4. Predict
    results = run_inference(model, inference_loader, CLASS_NAMES, device)
    
    # 5. Output Results
    print("\n--- INFERENCE RESULTS ---")
    for i, result in enumerate(results):
        print(f"Flow {i+1}: Predicted Traffic Type -> {result}")