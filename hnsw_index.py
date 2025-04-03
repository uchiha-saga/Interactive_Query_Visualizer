# custom_hnsw.py
import numpy as np
from supabase_setup import embeddings, model, texts

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

class CustomHNSW:
    def __init__(self, data, distance_metric):
        self.data = data  # NumPy array of embeddings
        self.distance_metric = distance_metric
        self.graph = self.build_graph(data)
    
    def build_graph(self, data):
        """
        Build a simple neighbor graph. For each node, we find the nearest 3 neighbors
        (excluding itself) and store them. This is a very simplified version and does not
        represent the full HNSW structure.
        """
        graph = {}
        for i, vec in enumerate(data):
            # Compute distances from vec to all nodes
            distances = np.linalg.norm(data - vec, axis=1)
            # Exclude self by taking indices starting at 1
            neighbor_indices = np.argsort(distances)[1:4]  # adjust number of neighbors as needed
            graph[i] = neighbor_indices.tolist()
        return graph

    def search(self, query_vector):
        # Choose a random entry node.
        entry_node = np.random.choice(len(self.data))
        current_node = entry_node
        best_distance = self.distance_metric(query_vector, self.data[current_node])
        traversal_log = []  # Log only improvements.
        
        improved = True
        while improved:
            improved = False
            for neighbor in self.graph[current_node]:
                d = self.distance_metric(query_vector, self.data[neighbor])
                if d < best_distance:
                    best_distance = d
                    current_node = neighbor
                    traversal_log.append(neighbor)
                    improved = True
                    break  # Move to the better neighbor immediately.
        return current_node, traversal_log, entry_node



# Prepare the data
data = np.array(embeddings)

# Create an instance of our custom HNSW with Euclidean distance.
custom_index = CustomHNSW(data, euclidean_distance)

def query_custom_hnsw(query_vector):
    nearest_idx, traversal_log, entry_node = custom_index.search(query_vector)
    print("Custom HNSW traversal log:", traversal_log)
    print("Entry node:", entry_node)
    print("Nearest neighbor index:", nearest_idx)
    return nearest_idx, traversal_log, entry_node


if __name__ == "__main__":
    test_query = "cat"
    # Get the embedding for the test query. Note: model.encode returns a 2D array.
    test_query_embed = model.encode([test_query], convert_to_numpy=True)[0]
    query_custom_hnsw(test_query_embed)
