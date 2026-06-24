#!/bin/bash

# Exit immediately if any command fails
set -e

# Check if the user provided a PCAP filename, otherwise default to Tor.pcap
INPUT_FILE=${1:-"Tor.pcap"}

# Strip the extension to create a clean output name (e.g., Tor.pcap -> Tor)
BASENAME="${INPUT_FILE%.*}"
OUTPUT_CSV="results_${BASENAME}.csv"

echo "============================================="
echo " BTRFormer Inference Pipeline Automation"
echo "============================================="
echo "Target PCAP : Pipeline/Input/$INPUT_FILE"
echo "Output CSV  : Pipeline/Input/$OUTPUT_CSV"
echo "---------------------------------------------"

# Step 1: Activate the virtual environment
echo "[*] Activating virtual environment (btr_env)..."
source btr_env/bin/activate

# Step 2: Run the Python pipeline
echo "[*] Initializing model and processing traffic..."
python Pipeline/pcap_btrformer.py \
    --pcap "Pipeline/Input/$INPUT_FILE" \
    --output "Pipeline/Input/$OUTPUT_CSV"

# Step 3: Deactivate the environment cleanly
deactivate

echo "---------------------------------------------"
echo "[+] Success! Pipeline completed."
echo "============================================="