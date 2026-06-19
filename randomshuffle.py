import os
import random
import shutil

# Root path where your processed data lives
base_dir = '/Users/raghavdutta/Documents/ETA/Model/BTRFormer/Dat/processed'
categories = ['Normal', 'Tor', 'VPN']

for cat in categories:
    train_path = os.path.join(base_dir, 'train', cat)
    test_path = os.path.join(base_dir, 'test', cat)
    
    # 1. Pull any existing files from test back into train
    if os.path.exists(test_path):
        for file in os.listdir(test_path):
            if file.endswith('.png'):
                shutil.move(os.path.join(test_path, file), os.path.join(train_path, file))
                
    # 2. Gather all images now back in the train folder
    all_images = [f for f in os.listdir(train_path) if f.endswith('.png')]
    total_count = len(all_images)
    
    # Calculate a clean 20% for testing
    test_count = int(total_count * 0.20)
    print(f"Category {cat}: Found {total_count} total images. Selecting {test_count} randomly for testing...")
    
    # 3. Shuffle the deck and pick random images
    random.seed(42)  # Ensures reproducibility
    random.shuffle(all_images)
    test_images = all_images[:test_count]
    
    # 4. Move the randomized selection into the test folder
    os.makedirs(test_path, exist_ok=True)
    for img in test_images:
        shutil.move(os.path.join(train_path, img), os.path.join(test_path, img))
        
print("Successfully created a randomized 80/20 train/test split!")