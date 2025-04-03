# visualize.py
import matplotlib.pyplot as plt
import numpy as np
from vector_ops import reduce_dimensions
from supabase_setup import embeddings, model  # reusing existing embeddings and model

def visualize_embeddings(embeddings_list, query_embedding=None):
    """
    Visualize embeddings with PCA and optionally highlight a query embedding.
    
    Args:
      embeddings_list : list or np.ndarray of embeddings.
      query_embedding : (optional) np.ndarray for the query.
    """
    embeddings_np = np.array(embeddings_list)
    
    # Reduce dimensions to 2D for plotting
    reduced = reduce_dimensions(embeddings_np, n_components=2)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], label='Data Embeddings', s=50, alpha=0.7)
    
    if query_embedding is not None:
        # Reduce the query vector too
        query_reduced = reduce_dimensions(query_embedding, n_components=2)
        plt.scatter(query_reduced[:, 0], query_reduced[:, 1], color='red', label='Query', s=100, marker='X')
    
    plt.legend()
    plt.title("PCA Visualization of Embeddings")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

if __name__ == "__main__":
    # Quick test visualization for a sample query "cat"
    sample_query = "cat"
    query_embed = model.encode([sample_query], convert_to_numpy=True)
    visualize_embeddings(embeddings, query_embedding=query_embed)
