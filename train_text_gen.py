import pandas as pd
import numpy as np
import re
import os
from sentence_transformers import SentenceTransformer


# -------------------------------
# Step 2: Load CSV
# -------------------------------
csv_path = "/content/drive/MyDrive/images/local_datasets/train_local.csv"
df = pd.read_csv(csv_path)

# -------------------------------
# Step 3: Extract structured fields
# -------------------------------
def parse_catalog(text):
    item_name = ""
    quantity = np.nan
    unit = ""

    if not isinstance(text, str):
        return item_name, quantity, unit

    # Fix encoding issues
    text = text.encode('utf-8', 'ignore').decode('utf-8')

    # Extract Item Name
    match = re.search(r"Item Name:\s*(.*)", text)
    if match:
        item_name = match.group(1).strip()

    # Extract Value
    match = re.search(r"Value:\s*([\d\.]+)", text)
    if match:
        quantity = float(match.group(1))

    # Extract Unit
    match = re.search(r"Unit:\s*(.*)", text)
    if match:
        unit = match.group(1).strip()

    return item_name, quantity, unit

df[['item_name', 'quantity', 'unit']] = df['catalog_content'].apply(
    lambda x: pd.Series(parse_catalog(x))
)

# -------------------------------
# Step 4: Text preprocessing
# -------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Fix encoding issues
    text = text.encode('utf-8', 'ignore').decode('utf-8')

    # Lowercase
    text = text.lower()

    # Replace newlines with spaces
    text = text.replace('\n', ' ').replace('\r', ' ')

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Remove special characters except alphanumerics and spaces
    text = re.sub(r"[^a-z0-9\s]", "", text)

    text = text.strip()
    return text

df['item_name_clean'] = df['item_name'].apply(clean_text)

# -------------------------------
# Step 5: Text embeddings
# -------------------------------
text_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim embeddings

text_embeddings = text_model.encode(
    df['item_name_clean'].tolist(),
    batch_size=64,
    show_progress_bar=True
)

# -------------------------------
# Step 6: Save embeddings
# -------------------------------
# Make folder if not exists
os.makedirs("/content/drive/MyDrive/images/text_embeddings", exist_ok=True)

text_emb_path = "/content/drive/MyDrive/images/text_embeddings/text_embeddings.npy"
np.save(text_emb_path, text_embeddings)

print("Text embeddings shape:", text_embeddings.shape)
print("Text embeddings saved at:", text_emb_path)