# Mount Drive

from google.colab import drive
drive.mount('/content/drive')

# ======================================================================
# The New, Improved, RESUMABLE CLIP Feature Extraction Script
# ======================================================================
import pandas as pd
import os
from pathlib import Path
import torch
import clip
from PIL import Image
import numpy as np
from tqdm import tqdm
import time

# -------------------------------
# Step 0: Install dependencies
# -------------------------------

!pip install -q ftfy regex tqdm
!pip install -q git+https://github.com/openai/CLIP.git

# -------------------------------
# Step 1: Settings and Paths
# -------------------------------
print("Setting up paths for the TEST dataset...")

CHUNK_SIZE = 5000 # We will process and save in chunks of 5000 images

csv_path = "/content/drive/MyDrive/images/test_local.csv"
df = pd.read_csv(csv_path)
image_files = df['local_image_path'].tolist()

# We will save to the same folder as your teammate for consistency
save_folder = "/content/drive/MyDrive/images/image_embeddings"
os.makedirs(save_folder, exist_ok=True)

# --- NEW: We'll create a temporary folder for our chunks ---
chunk_folder = os.path.join(save_folder, "test_chunks")
os.makedirs(chunk_folder, exist_ok=True)

final_emb_path = os.path.join(save_folder, "test_image_features_clip.npy")
print(f"Temporary chunks will be saved in: {chunk_folder}")
print(f"Final embeddings will be saved to: {final_emb_path}")

# -------------------------------
# Step 2: Load CLIP model
# -------------------------------
print("\nLoading CLIP model ViT-B/32...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# -------------------------------
# Step 3: Extract Embeddings in Chunks (The "Smart" Part)
# -------------------------------
num_chunks = (len(image_files) + CHUNK_SIZE - 1) // CHUNK_SIZE

print(f"\nFound {len(image_files)} images to process. This will be done in {num_chunks} chunks.")

for i in range(num_chunks):
    chunk_start = i * CHUNK_SIZE
    chunk_end = min((i + 1) * CHUNK_SIZE, len(image_files))
    chunk_files = image_files[chunk_start:chunk_end]

    # --- THIS IS THE RESUME LOGIC ---
    chunk_save_path = os.path.join(chunk_folder, f"chunk_{i:02d}.npy")
    if os.path.exists(chunk_save_path):
        print(f"Chunk {i+1}/{num_chunks} already exists. Skipping.")
        continue

    print(f"Processing Chunk {i+1}/{num_chunks} ({len(chunk_files)} images)...")

    chunk_embeddings = []
    for img_path in tqdm(chunk_files, desc=f"Chunk {i+1}"):
        if isinstance(img_path, str) and os.path.exists(img_path):
            try:
                image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model.encode_image(image)
                    emb = emb.cpu().numpy().flatten()
                chunk_embeddings.append(emb)
            except Exception as e:
                chunk_embeddings.append(np.zeros(512))
        else:
            chunk_embeddings.append(np.zeros(512))

    # Save the processed chunk immediately
    np.save(chunk_save_path, np.vstack(chunk_embeddings))
    print(f"SUCCESS: Saved {chunk_save_path}")

# -------------------------------
# Step 4: Stitch All Chunks Together
# -------------------------------
print("\nAll chunks processed. Stitching them together into the final file...")

all_embeddings = []
for i in range(num_chunks):
    chunk_path = os.path.join(chunk_folder, f"chunk_{i:02d}.npy")
    chunk_data = np.load(chunk_path)
    all_embeddings.append(chunk_data)

final_embeddings = np.vstack(all_embeddings)
np.save(final_emb_path, final_embeddings)

print("="*40)
print("SUCCESS! Final TEST CLIP image embeddings saved at:", final_emb_path)
print("Final embeddings shape:", final_embeddings.shape)
print("="*40)