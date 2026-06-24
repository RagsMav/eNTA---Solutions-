# How to Train Yourself
<br>

Open a terminal window and navigate to BTRFormer repo ( after copying this repo)
<br>
```
mkdir Data
cd Data
mkdir raw
cd raw
mkdir Normal
mkdir Tor
mkdir VPN
```
<br>

Run these commands one after another
<br>
Now update the paths in the following files
<br>
1) in data_generation.py -> root_dir at the end of the file -> to your path for the Data folder
2) in config.py -> _C.DATA.DATA_PATH -> to your path
<br>
When adding files make sure to name them as exactly "profile_1_Tor", "profile_2_VPN" and "profile_3_Normal" and their extensions pcap 
add these pcaps to respective folders in the Document.

Here only 3 classes because in config.py
```
_C.MODEL.NUM_CLASSES = 3
```
This was set to 3, it can be changed to any number provided you have sufficient label data
<br>

Navigate using cd to the data process folder
Run the data_generation.py folder
<br>
##### Important - (if running on linux then only run this before running data_generation.py)
```
sudo apt update
sudo apt install mono-complete
```
After this run
```
python3 data_generation.py
```
This should split your pcaps and create a splitcap folder, where your 1 large pcap would be split into flows
<br>
Create the train and test folders
```
mkdir BTRFormer/Data/processed
cd BTRFormer/Data/processed
mkdir train
mkdir test
```
Now run the BTR_generator.py
```
# 1. Force create the exact directories so Linux can't complain
mkdir -p BTRFormer/Data/processed/train/Normal
mkdir -p BTRFormer/Data/processed/train/Tor
mkdir -p BTRFormer/Data/processed/train/VPN

# 2. Run the script pointing to the correct 'Data' output folder
python BTR_generator.py --flows_pcap_path /BTRFormer/Data/splitcap/ --output_dir /BTRFormer/Data/processed/train/
```

Instead of the BTRFormer/Data/processed/train/. write write the full path, where you want to create the 3 processed folders
<br>
Move the balance_and_split.py to processed folder
<br>
Now run the balance_and_split.py to make the data even
```
python balance_and_split.py --train train --test test --delete-excess
```

Instead of the BTRFormer/Data/processed/train/. write write the full path, where you want to create the 3 processed folders
<br>
Now run the Bash script run_btrformer_training.bash.
<br>
I have uploaded a hugging face model link
```
https://huggingface.co/RagsMav/eNTA-Solutions/tree/main
```

### Only for Tor, VPN and Normal Classification
Uplaod your .pth model in Pipeline/output/btrformer/default
<br>
If a folder named output doesnt exist make the entire output then btrformer inside it and then default inside it
<br>
Upload the file to analyze in the Pipeline/Input folder
<br>
Now go into the Pipeline folder from the terminal using cd.
```
python pcap_btrformer.py --pcap Input/Tor.pcap --output Input/results.csv
```
Run this and wait some seconds it will give you the CSV which has the confidence scores
<br>
#### How to run the bash file run_inference

```
chmod +x run_inference.bash
./run_inference.bash your_capture.pcap
```
