from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import plotly.graph_objects as go  # optional for debugging

import os
import time
import random
from glove_loader import load_glove_embeddings
from hnsw_index import CompleteHNSW
from acorn1_index import ACORN1
import numpy as np

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)  # Allow frontend requests

# Load GloVe
texts, vectors = load_glove_embeddings("glove.6B.100d.txt", max_words=2500)

# Normalize GloVe vectors
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

# Load SBERT
model = SentenceTransformer("all-MiniLM-L6-v2")

# Fit SBERT → GloVe space PCA (384 → 100)
print("⏳ Fitting PCA: SBERT → GloVe")
sbert_embeddings = model.encode(texts[:1000])  # Use first 1000 GloVe words
sbert_to_glove_pca = PCA(n_components=100)
sbert_to_glove_pca.fit(sbert_embeddings)

# Fit 3D PCA on GloVe for visualization
pca = PCA(n_components=3)
reduced_embeddings = pca.fit_transform(vectors)

# Build HNSW and ACORN
hnsw_index = CompleteHNSW(vectors, M=10)
acorn_index = ACORN1(hnsw_index)


def get_query_vector(word):
    if word in texts:
        idx = texts.index(word)
        return vectors[idx], word, idx
    else:
        print(f"'{word}' not found. Using SBERT to find similar match...")
        query_embed = model.encode([word])[0].reshape(1, -1)

        # Reduce SBERT 384D → GloVe 100D
        reduced_query = sbert_to_glove_pca.transform(query_embed)
        reduced_query /= np.linalg.norm(reduced_query)

        sims = np.dot(vectors, reduced_query[0])
        idx = np.argmax(sims)
        print(f"Closest match in GloVe: '{texts[idx]}'")
        return vectors[idx], texts[idx], idx



@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)


@app.route("/query", methods=["POST"])
def query():
    try:
        data = request.get_json()
        word = data.get("word", "")
        query_vector, actual_word, query_idx = get_query_vector(word)

        # Run HNSW and ACORN searches
        start_hnsw = time.time()
        hnsw_result, hnsw_log, entry_node = hnsw_index.search(query_vector)
        end_hnsw = time.time()

        start_acorn = time.time()
        acorn_result, acorn_path, _ = acorn_index.search(query_vector, start_node=entry_node)
        end_acorn = time.time()

        # Reduce query to 3D
        query_3d = pca.transform(query_vector.reshape(1, -1))[0].tolist()

        # Collect all visited nodes
        visited_nodes = set([i for path in hnsw_log.values() for i in path] + acorn_path)
        node_positions = {
            str(i): reduced_embeddings[i].tolist() for i in visited_nodes
        }

        layer_assignments = {}
        for layer, nodes in hnsw_index.layers.items():
            for n in nodes:
                layer_assignments[str(n)] = layer

        def cosine_sim(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        return jsonify({
            "query": actual_word,
            "query_coords": query_3d,
            "entry_point": entry_node,
            "entry_coords": reduced_embeddings[entry_node].tolist(),
            "hnsw": {
                "result": texts[int(hnsw_result)],
                "path": {str(int(k)): [int(x) for x in v] for k, v in hnsw_log.items()},
                "time_ms": round((end_hnsw - start_hnsw) * 1000, 2),
                "num_visited": int(sum(len(v) for v in hnsw_log.values())),
                "similarity": cosine_sim(query_vector, vectors[int(hnsw_result)])
            },
            "acorn": {
                "result": texts[int(acorn_result)],
                "path": [int(x) for x in acorn_path],
                "time_ms": round((end_acorn - start_acorn) * 1000, 2),
                "num_visited": int(len(acorn_path)),
                "similarity": cosine_sim(query_vector, vectors[int(acorn_result)])
            },
            "positions": {str(i): reduced_embeddings[i].tolist() for i in range(len(vectors))},
            "layers": {
                str(n): layer for layer, nodes in hnsw_index.layers.items() for n in nodes
            },
            "labels": {str(i): texts[i] for i in range(len(vectors))}

        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    app.run(debug=True)
