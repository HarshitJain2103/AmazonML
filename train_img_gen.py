# -------------------------------
# Step 1: Imports
# -------------------------------
import torch
import clip
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import time

# -------------------------------
# Step 2: Paths
# -------------------------------
csv_path = "/content/drive/MyDrive/images/local_datasets/train_local.csv"
df = pd.read_csv(csv_path)
image_folder = "/content/drive/MyDrive/images/train"

# Output folder
os.makedirs("/content/drive/MyDrive/images/image_embeddings", exist_ok=True)
emb_save_path = "/content/drive/MyDrive/images/image_embeddings/train_image_embeddings.npy"
progress_path = "/content/drive/MyDrive/images/image_embeddings/progress_log.txt"
failed_path = "/content/drive/MyDrive/images/image_embeddings/failed_images.txt"

# -------------------------------
# Step 3: Load CLIP model
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# -------------------------------
# Step 4: Prepare file list
# -------------------------------
image_files = [os.path.join(image_folder, f"{sid}.jpg") for sid in df["sample_id"].astype(str)]

# Resume from existing progress
if os.path.exists(emb_save_path):
    existing_embeddings = np.load(emb_save_path)
    start_idx = existing_embeddings.shape[0]
    image_embeddings = existing_embeddings.tolist()
    print(f"🔁 Resuming from index {start_idx}...")
else:
    image_embeddings = []
    start_idx = 0

failed_images = []

# -------------------------------
# Step 5: Embedding extraction with retry & checkpoint
# -------------------------------
for idx in tqdm(range(start_idx, len(image_files)), desc="CLIP image embeddings"):
    img_path = image_files[idx]
    for attempt in range(3):  # retry up to 3 times
        try:
            image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(image).cpu().numpy().flatten()
            image_embeddings.append(emb)
            break
        except Exception as e:
            print(f"⚠️ Error {img_path} (Attempt {attempt+1}/3): {e}")
            time.sleep(1)
    else:
        image_embeddings.append(np.zeros(512))  # fallback for failed images
        failed_images.append(img_path)

    # Save progress every 500 images or at the end
    if (idx + 1) % 500 == 0 or (idx + 1) == len(image_files):
        np.save(emb_save_path, np.vstack(image_embeddings))
        with open(progress_path, "w") as f:
            f.write(str(idx + 1))
        torch.cuda.empty_cache()
        os.sync()  # force write to disk
        print(f"💾 Checkpoint saved at index {idx+1}")

# -------------------------------
# Step 6: Final save
# -------------------------------
final_embeddings = np.vstack(image_embeddings)
np.save(emb_save_path, final_embeddings)
os.sync()

print("✅ All embeddings saved at:", emb_save_path)
print("Final shape:", final_embeddings.shape)

# -------------------------------
# Step 7: Log failed images (if any)
# -------------------------------
if failed_images:
    with open(failed_path, "w") as f:
        f.write("\n".join(failed_images))
    print(f"⚠️ Logged {len(failed_images)} failed images to:", failed_path)
else:
    print("🎉 No failed images!")