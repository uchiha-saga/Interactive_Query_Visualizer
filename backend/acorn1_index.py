import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from tqdm import tqdm


class ACORN1:
    def __init__(self, hnsw_index, radius=0.5, max_neighbors=20):
        """
        ACORN-1 reuses HNSW graph structure and augments it by adding local radius-based refinement.
        
        Parameters:
        - hnsw_index: CompleteHNSW instance
        - radius: Cosine similarity threshold for local neighborhood expansion
        - max_neighbors: Maximum number of neighbors to consider for local expansion
        """
        self.hnsw = hnsw_index
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.data = hnsw_index.data
        self.num_nodes = len(self.data)

        # Augmented neighborhood graph
        self.acorn_graph = self._augment_with_radius_neighbors()

    def _augment_with_radius_neighbors(self):
        """
        For each node, expand its neighborhood with vectors within a cosine distance radius.
        """
        graph = defaultdict(set)
        for idx in tqdm(range(self.num_nodes), desc="🔧 Building ACORN-1 Neighborhoods"):
            vec = self.data[idx].reshape(1, -1)
            sims = cosine_similarity(vec, self.data)[0]
            neighbor_indices = np.where(sims >= (1 - self.radius))[0]

            # Limit to max_neighbors
            neighbor_indices = sorted(neighbor_indices, key=lambda i: -sims[i])
            for neighbor in neighbor_indices[:self.max_neighbors]:
                if neighbor != idx:
                    graph[idx].add(neighbor)
                    graph[neighbor].add(idx)  # Ensure bidirectional links

        return {k: list(v) for k, v in graph.items()}

    def search(self, query_vector, start_node=None, force_best_entry=True):
        query_vector = query_vector / np.linalg.norm(query_vector)
        if start_node is None:
            start_node = self.hnsw.entry_point

        visited = set()
        current_node = start_node
        path = [current_node]
        improved = True

        while improved:
            improved = False
            visited.add(current_node)
            current_vec = self.data[current_node]
            current_dist = 1 - np.dot(query_vector, current_vec)

            # ACORN-1: Expand to 2-hop neighbors
            neighbors = set(self.acorn_graph.get(current_node, []))
            for n in list(neighbors):
                neighbors.update(self.acorn_graph.get(n, []))  # 2-hop

            best_neighbor = None
            best_dist = current_dist

            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                neighbor_vec = self.data[neighbor]
                neighbor_dist = 1 - np.dot(query_vector, neighbor_vec)
                if neighbor_dist < best_dist:
                    best_neighbor = neighbor
                    best_dist = neighbor_dist

            if best_neighbor is not None:
                current_node = best_neighbor
                path.append(current_node)
                improved = True

        return current_node, path, start_node

                