import numpy as np
import random
from collections import defaultdict
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class CompleteHNSW:
    def __init__(self, data, M=10, middle_ratio=0.1, entry_ratio=0.1, n_jobs=-1):
        self.data = np.array(data, dtype=np.float32)
        self.data /= np.linalg.norm(self.data, axis=1, keepdims=True)  # Normalize for cosine

        self.num_nodes = len(self.data)
        self.M = M
        self.n_jobs = n_jobs  # Number of parallel jobs (CPU cores)

        # Build hierarchical layers
        all_indices = list(range(self.num_nodes))
        self.layers = {
            0: all_indices,
            1: random.sample(all_indices, max(1, round(self.num_nodes * middle_ratio)))
        }
        self.layers[2] = random.sample(
            self.layers[1], max(1, round(len(self.layers[1]) * entry_ratio))
        )

        if not self.layers[2]:
            raise ValueError("Entry layer is empty! Increase entry_ratio or middle_ratio.")

        # Build graph for each layer in parallel
        self.graphs = {}
        for layer, indices in self.layers.items():
            print(f"⏳ Building graph for layer {layer} with {len(indices)} nodes...")
            graph = self._build_layer_graph(indices)
            self.graphs[layer] = graph

        # Randomly select entry point from top layer
        self.entry_point = random.choice(self.layers[2])

    def _build_layer_graph(self, indices):
        """Builds a bidirectional graph for one layer using cosine similarity and parallelism."""
        data_layer = self.data[indices]

        def process_node(i):
            vec_i = data_layer[i].reshape(1, -1)
            sims = cosine_similarity(vec_i, data_layer)[0]
            sims[i] = -1  # Exclude self
            top_k = np.argpartition(sims, -self.M)[-self.M:]
            return i, top_k.tolist()

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(process_node)(i) for i in tqdm(range(len(indices)), desc="  ↳ Connecting nodes", leave=False)
        )

        graph = defaultdict(set)
        index_map = {local_idx: global_idx for local_idx, global_idx in enumerate(indices)}

        for local_i, neighbor_ids in results:
            global_i = index_map[local_i]
            for local_j in neighbor_ids:
                global_j = index_map[local_j]
                graph[global_i].add(global_j)
                graph[global_j].add(global_i)  # Bidirectional

        return {k: list(v) for k, v in graph.items()}

    def search(self, query_vector):
        """Greedy layer-wise HNSW search using cosine similarity (assumes normalized vectors)."""
        query_vector = query_vector / np.linalg.norm(query_vector)  # Normalize query
        traversal_log = {}
        current_node = self.entry_point

        for layer in sorted(self.layers.keys(), reverse=True):
            traversal_log[layer] = [current_node]
            improved = True
            while improved:
                improved = False
                for neighbor in self.graphs[layer].get(current_node, []):
                    d_curr = 1 - np.dot(query_vector, self.data[current_node])
                    d_neighbor = 1 - np.dot(query_vector, self.data[neighbor])
                    if d_neighbor < d_curr:
                        current_node = neighbor
                        traversal_log[layer].append(neighbor)
                        improved = True
                        break

        return current_node, traversal_log, self.entry_point
