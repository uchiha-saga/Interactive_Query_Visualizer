
import db_config
from sentence_transformers import SentenceTransformer
from supabase import create_client

url = db_config.DB_HOST
key = db_config.DB_ANON
supabase = create_client(url, key)

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = ["cat", "dog", "fish"]
embeddings = model.encode(texts).tolist()

for text, embed in zip(texts, embeddings):
    response = supabase.table("items").insert({
        "description": text,
        "embedding": embed
    }).execute()
    print("Inserted:", response.data)

