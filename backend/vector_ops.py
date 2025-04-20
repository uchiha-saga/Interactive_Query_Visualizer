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
    embeddings = np.atleast_2d(embeddings)  # Ensures correct shape
    embeddings_norm = np.linalg.norm(embeddings, axis=1)

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

def get_pca_info(data, n_components=3):
    """
    Computes how much variance is retained after PCA reduction.

    Parameters:
        data : np.ndarray of shape (n_samples, n_features)
        n_components : int, number of PCA dimensions to keep

    Returns:
        float: Fraction of variance retained (e.g., 0.823 means 82.3% retained)
    """
    try:
        data = np.array(data, dtype=np.float64)
        actual_components = min(n_components, data.shape[1], data.shape[0])
        pca = PCA(n_components=actual_components)
        pca.fit(data)
        return np.sum(pca.explained_variance_ratio_)
    except Exception as e:
        print(f"Error in computing PCA info: {str(e)}")
        print(traceback.format_exc())
        return -1.0  # or raise an error if preferred

def normalize_vectors(vectors):
    """
    Normalize vectors to unit length along axis=1 (rows).

    Args:
        vectors (np.ndarray): Shape (n_samples, n_features)

    Returns:
        np.ndarray: Normalized vectors
    """
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def get_query_embedding(word, model, glove_vectors, glove_words, pca_model):
    """
    Return vector, resolved word, and index in GloVe space.

    If the word is in GloVe, returns that.
    Otherwise uses SBERT + PCA to find closest match.

    Returns:
        (np.ndarray, str, int): vector, word, index
    """
    if word in glove_words:
        idx = glove_words.index(word)
        return glove_vectors[idx], word, idx
    else:
        print(f"'{word}' not in GloVe, falling back to SBERT...")
        query_embed = model.encode([word])[0].reshape(1, -1)
        reduced_query = pca_model.transform(query_embed)
        reduced_query /= np.linalg.norm(reduced_query)
        sims = np.dot(glove_vectors, reduced_query[0])
        idx = np.argmax(sims)
        print(f"Closest GloVe match: '{glove_words[idx]}'")
        return glove_vectors[idx], glove_words[idx], idx


def fit_sbert_to_glove_pca(model, glove_words, glove_dim=100, max_words=1000):
    sbert_embeddings = model.encode(glove_words[:max_words])
    pca = PCA(n_components=glove_dim)
    pca.fit(sbert_embeddings)
    return pca
