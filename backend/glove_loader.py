# glove_loader.py
import numpy as np
from tqdm import tqdm

def load_glove_embeddings(file_path, max_words=None):
    """
    Load GloVe embeddings from a file.
    
    Args:
        file_path (str): Path to the GloVe .txt file.
        max_words (int, optional): Limit to first N words for faster testing.
        
    Returns:
        Tuple[List[str], np.ndarray]: Words and their corresponding vectors
    """
    words = []
    vectors = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in tqdm(enumerate(f), desc="Loading GloVe embeddings", total=max_words or 0):
            if max_words and i >= max_words:
                break
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            words.append(word)
            vectors.append(vector)

    print(1)
    vectors = np.vstack(vectors)

    print(2)

    print(len(words), vectors.shape)
    #return None, None
    return words, vectors
