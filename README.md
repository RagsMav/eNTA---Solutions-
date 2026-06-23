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
when adding files make sure to name them as exactly "profile_1_Tor", "profile_2_VPN" and "profile_3_Normal" and their extensions pcap 
add these pcaps to respective folders in the Document.

Here only 3 classes because in config.py
```
_C.MODEL.NUM_CLASSES = 3
```
This was set to 3, it can be changed to any number provided you have sufficient label data
<br>

navigate using cd to the data process folder
run the data_generation.py folder
<br>
##### important - (if running on linux then only run this before running data_generation.py)
```
sudo apt update
sudo apt install mono-complete
```
After this run
```
python3 data_generation.py
```
This should create split your pcaps and create a splitcap folder, where your 1 large pcap would be split into flows
<br>

The BTR_generator.py


