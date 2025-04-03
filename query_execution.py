# query_execution.py
from supabase_setup import model, embeddings, texts
from hnsw_index import query_custom_hnsw  # Make sure to import the custom function
import numpy as np
from vector_ops import compute_cosine_similarity, compute_euclidean_distance

def execute_query(query_text, k=3):
    query_embed = model.encode([query_text], convert_to_numpy=True)
    nearest_idx, traversal_log, entry_node = query_custom_hnsw(query_embed[0])
    embeddings_np = np.array(embeddings)
    cosine_sim = compute_cosine_similarity(query_embed[0], embeddings_np)
    euclidean_dist = compute_euclidean_distance(query_embed[0], embeddings_np)
    return {
        "nearest_idx": nearest_idx,
        "traversal_log": traversal_log,
        "entry_node": entry_node,
        "cosine_similarity": cosine_sim,
        "euclidean_distance": euclidean_dist
    }


if __name__ == "__main__":
    query_text = input("Enter your query: ")
    result = execute_query(query_text, k=3)
    print("\nExecution Result:", result)
