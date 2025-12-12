import os
import pandas as pd
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ============================================================
# Config (Environment Independent)
# ============================================================

PARQUET_PATH = "./agents/tools/data/products.parquet"
PERSIST_DIR = "./agents/tools/data/chroma_toys"
COLLECTION_NAME = "products_toys"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

def build_features_column(df: pd.DataFrame) -> pd.Series:
    """
    Build the 'features' text field used for embeddings.
    """
    if "features" in df.columns:
        return df["features"].fillna("")

    # Try to be robust to different column names
    title = df.get("title", pd.Series([""] * len(df))).fillna("")
    brand = df.get("brand", pd.Series([""] * len(df))).fillna("")
    category = df.get("category", pd.Series([""] * len(df))).fillna("")
    ingredients = df.get("ingredients", pd.Series([""] * len(df))).fillna("")
    
    # Handle rating safely
    if "rating" in df.columns:
        rating = df["rating"].astype(str).fillna("")
    else:
        rating = pd.Series([""] * len(df))

    description = df.get("description", pd.Series([""] * len(df))).fillna("")
    about = df.get("about", pd.Series([""] * len(df))).fillna("")
    spec = df.get("specification", pd.Series([""] * len(df))).fillna("")

    # Concatenate features
    features = (
        title + " " +
        brand + " " +
        category + " " +
        ingredients + " " +
        rating + " " +
        description + " " +
        about + " " +
        spec
    )

    return features

def main():
    # ========================================================
    # 0. Setup & Check
    # ========================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using Device: {device}")

    if not os.path.exists(PARQUET_PATH):
        print(f"❌ Error: Data file not found at {PARQUET_PATH}")
        print("Please create a 'data' folder and put 'products.parquet' inside it.")
        return

    # ========================================================
    # 1. Load products parquet
    # ========================================================
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded products parquet: {df.shape}")
    
    # Ensure numeric types for price/rating
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df[df["price"].notna()] # Drop invalid prices
    
    if "rating" not in df.columns:
        df["rating"] = None
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # ========================================================
    # 2. Build features column for embeddings
    # ========================================================
    print("Building feature strings...")
    df["features"] = build_features_column(df)

    # ========================================================
    # 3. Initialize embedding model & Chroma
    # ========================================================
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

    os.makedirs(PERSIST_DIR, exist_ok=True)

    client = chromadb.PersistentClient(
        path=PERSIST_DIR,
        settings=Settings(allow_reset=True)
    )

    print("Resetting Database...")
    client.reset() 

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Toys/products catalog with rating & ingredients"}
    )

    # ========================================================
    # 4. Prepare data for indexing
    # ========================================================
    texts = df["features"].tolist()
    ids = df["id"].astype(str).tolist()

    metadatas = []
    for _, row in df.iterrows():
        meta = {
            "id": str(row["id"]),
            "title": str(row.get("title", "")),
            "brand": str(row.get("brand", "")),
            "category": str(row.get("category", "")),
            "price": float(row["price"]) if pd.notna(row["price"]) else 0.0,
            "rating": float(row["rating"]) if pd.notna(row["rating"]) else 0.0,
            "ingredients": str(row.get("ingredients", "")),
        }
        metadatas.append(meta)

    print(f"Number of items to index: {len(ids)}")

    # ========================================================
    # 5. Index in batches
    # ========================================================
    BATCH_SIZE = 256

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Indexing"):
        end = start + BATCH_SIZE
        batch_texts = texts[start:end]
        batch_ids = ids[start:end]
        batch_metas = metadatas[start:end]

        batch_embeds = model.encode(batch_texts, show_progress_bar=False).tolist()

        collection.add(
            ids=batch_ids,
            embeddings=batch_embeds,
            metadatas=batch_metas,
            documents=batch_texts
        )

    print("✅ Indexing complete.")
    print(f"Chroma persisted at: {PERSIST_DIR}")
    print(f"Collection name: {COLLECTION_NAME}")

if __name__ == "__main__":
    main()