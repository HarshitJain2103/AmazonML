import pandas as pd
import numpy as np
import re
import os
from sentence_transformers import SentenceTransformer

print("Generating Text Embeddings for the TEST set using all-MiniLM-L6-v2")
print("="*70)

# -------------------------------
# Step 1: Load the CORRECT TEST CSV
# -------------------------------
# --- THIS PATH IS NOW CORRECTED TO MATCH YOUR TEAM'S FOLDER STRUCTURE ---
csv_path = "/content/drive/MyDrive/images/local_datasets/test_local.csv"
print(f"Loading data from: {csv_path}")
df = pd.read_csv(csv_path)

# -------------------------------
# Step 2: Replicate the EXACT same parsing logic
# -------------------------------
def parse_catalog(text):
    item_name = ""
    quantity = np.nan
    unit = ""
    if not isinstance(text, str):
        return item_name, quantity, unit
    text = text.encode('utf-8', 'ignore').decode('utf-8')
    match = re.search(r"Item Name:\s*(.*)", text)
    if match:
        item_name = match.group(1).strip()
    match = re.search(r"Value:\s*([\d\.]+)", text)
    if match:
        quantity = float(match.group(1))
    match = re.search(r"Unit:\s*(.*)", text)
    if match:
        unit = match.group(1).strip()
    return item_name, quantity, unit

print("Parsing catalog content to extract item names...")
df[['item_name', 'quantity', 'unit']] = df['catalog_content'].apply(
    lambda x: pd.Series(parse_catalog(x)))

# -------------------------------
# Step 3: Replicate the EXACT same text cleaning logic
# -------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.encode('utf-8', 'ignore').decode('utf-8')
    text = text.lower()
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = text.strip()
    return text

print("Cleaning extracted item names...")
df['item_name_clean'] = df['item_name'].apply(clean_text)

# -------------------------------
# Step 4: Generate text embeddings using the SAME model
# -------------------------------
print("\nLoading model 'all-MiniLM-L6-v2'...")
text_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Generating text embeddings for the TEST set (this will take 20-30 mins)...")
text_embeddings = text_model.encode(
    df['item_name_clean'].tolist(),
    batch_size=64,
    show_progress_bar=True
)

# -------------------------------
# Step 5: Save embeddings with a clear, specific name
# -------------------------------
# Let's save to the team's 'text_embeddings' folder for consistency
save_folder = "/content/drive/MyDrive/images/text_embeddings"
os.makedirs(save_folder, exist_ok=True)

# We name the file clearly to show it's for the TEST set
text_emb_path = os.path.join(save_folder, "test_text_features_minilm.npy")
np.save(text_emb_path, text_embeddings)

print("\nSUCCESS!")
print("Text embeddings shape:", text_embeddings.shape)
print("Text embeddings for TEST set saved at:", text_emb_path)
print("="*70)