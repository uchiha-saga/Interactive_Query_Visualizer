# supabase_setup.py
from supabase import create_client
from sentence_transformers import SentenceTransformer
import db_config as db_config  # Contains DB_HOST and DB_ANON

# Connect to Supabase
url = db_config.DB_HOST
key = db_config.DB_ANON
supabase = create_client(url, key)

# Clear previous data from the "items" table
delete_response = supabase.table("items").delete().gt("id", 0).execute()
print("Cleared previous data:", delete_response.data)

# Initialize the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Example texts and their embeddings
texts = ["cat", "dog", "fish", "tiger", "capibara", "monkey"]
embeddings = model.encode(texts, convert_to_numpy=True).tolist()

# Insert new embeddings into the "items" table
for text, embed in zip(texts, embeddings):
    response = supabase.table("items").insert({
        "description": text,
        "embedding": embed
    }).execute()
    print("Inserted:", response.data)

if __name__ == "__main__":
    print("Embedding for first text:", embeddings[0])
