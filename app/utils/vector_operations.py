import numpy as np
from sklearn.decomposition import PCA
import traceback

def reduce_dimensions(embeddings, n_components=2):
    """
    Reduce the dimensionality of embeddings using PCA.
    
    Args:
        embeddings (np.ndarray): Array of embeddings with shape (n_samples, n_features)
        n_components (int): Number of dimensions to reduce to
        
    Returns:
        np.ndarray: Reduced embeddings with shape (n_samples, n_components)
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(embeddings)

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

def reduce_dimensions(data, method='pca', n_dimensions=3):
    """
    Reduce dimensions of the input data for visualization
    
    Parameters:
    -----------
    data : numpy.ndarray
        High-dimensional embeddings
    method : str
        Dimensionality reduction method ('pca' or 'umap')
    n_dimensions : int
        Number of dimensions for the reduced data (default: 3 for 3D visualization)
        
    Returns:
    --------
    numpy.ndarray
        Reduced dimensionality data
    """
    try:
        if data is None or data.shape[0] == 0:
            # Return empty array if no data
            return np.array([])
        
        # Make sure data is the right format
        data = np.array(data, dtype=np.float64)
        
        # Handle case where dimensions are less than requested
        actual_n_dimensions = min(n_dimensions, data.shape[1], data.shape[0])
        
        if method == 'pca':
            pca = PCA(n_components=actual_n_dimensions)
            return pca.fit_transform(data)
        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=actual_n_dimensions)
                return reducer.fit_transform(data)
            except ImportError:
                print("UMAP not available, falling back to PCA")
                pca = PCA(n_components=actual_n_dimensions)
                return pca.fit_transform(data)
        else:
            print(f"Unsupported method '{method}', falling back to PCA")
            pca = PCA(n_components=actual_n_dimensions)
            return pca.fit_transform(data)
    except Exception as e:
        print(f"Error in dimension reduction: {str(e)}")
        print(traceback.format_exc())
        # Return 3D random data as fallback
        return np.random.rand(data.shape[0], 3) 