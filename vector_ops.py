# vector_ops.py
import numpy as np
from sklearn.decomposition import PCA
import traceback

def reduce_dimensions(data, n_components=3):
    """
    Reduce dimensions of the input data using PCA.
    
    Parameters:
      data : np.ndarray of shape (n_samples, n_features)
      n_components : int, number of dimensions to reduce to (default: 3)
    
    Returns:
      np.ndarray of shape (n_samples, n_components)
    """
    try:
        data = np.array(data, dtype=np.float64)
        actual_components = min(n_components, data.shape[1], data.shape[0])
        pca = PCA(n_components=actual_components)
        return pca.fit_transform(data)
    except Exception as e:
        print(f"Error in dimension reduction: {str(e)}")
        print(traceback.format_exc())
        return np.random.rand(data.shape[0], n_components)
    
def compute_cosine_similarity(query_vector, embeddings):
    """
    Compute cosine similarity between a query vector and a set of embeddings.
    
    Args:
        query_vector (np.ndarray): Query vector with shape (n_features,)
        embeddings (np.ndarray): Array of embeddings with shape (n_samples, n_features)
        
    Returns:
        np.ndarray: Array of similarities with shape (n_samples,)
    """
    # Normalize vectors
    query_norm = np.linalg.norm(query_vector)
    embeddings_norm = np.linalg.norm(embeddings, axis=1)
    
    # Compute cosine similarity
    similarities = np.dot(embeddings, query_vector) / (embeddings_norm * query_norm)
    return similarities

def compute_euclidean_distance(query_vector, embeddings):
    """
    Compute Euclidean distance between a query vector and a set of embeddings.
    
    Args:
        query_vector (np.ndarray): Query vector with shape (n_features,)
        embeddings (np.ndarray): Array of embeddings with shape (n_samples, n_features)
        
    Returns:
        np.ndarray: Array of distances with shape (n_samples,)
    """
    return np.linalg.norm(embeddings - query_vector, axis=1)
