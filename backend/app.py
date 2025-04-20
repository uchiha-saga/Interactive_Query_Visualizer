from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import numpy as np
import time

from glove_loader import load_glove_embeddings
from hnsw_index import CompleteHNSW
from acorn1_index import ACORN1
from vector_ops import (
    reduce_dimensions,
    compute_cosine_similarity,
    normalize_vectors,
    get_query_embedding,
    fit_sbert_to_glove_pca
)

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)  # Allow frontend requests

# ========== Load & Preprocess ==========
print("Loading GloVe embeddings...")
texts, vectors = load_glove_embeddings("glove.6B.100d.txt", max_words=2500)
vectors = normalize_vectors(vectors)

print(" Loading SBERT model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print(" Fitting SBERT → GloVe PCA...")
sbert_to_glove_pca = fit_sbert_to_glove_pca(model, texts, glove_dim=100)

print(" Reducing GloVe to 3D for visualization...")
pca_3d = reduce_dimensions(vectors, n_components=3)

print(" Building HNSW and ACORN-1 indexes...")
hnsw_index = CompleteHNSW(vectors, M=10)
acorn_index = ACORN1(hnsw_index)

search_log = []  # Stores all searched words + results


# ========== Routes ==========

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

        query_vector, actual_word, query_idx = get_query_embedding(
            word, model, vectors, texts, sbert_to_glove_pca
        )


        # HNSW Search
        start_hnsw = time.time()
        hnsw_result, hnsw_log, entry_node = hnsw_index.search(query_vector)
        end_hnsw = time.time()

        # ACORN-1 Search
        start_acorn = time.time()
        acorn_result, acorn_path, _ = acorn_index.search(query_vector, start_node=entry_node)
        end_acorn = time.time()

        # Reduce query vector to 3D
        query_3d = pca_3d[query_idx].tolist()

        # Visited nodes
        visited_nodes = set([i for path in hnsw_log.values() for i in path] + acorn_path)
        node_positions = {
            str(i): pca_3d[i].tolist() for i in visited_nodes
        }

        # Layer assignment for nodes
        layer_assignments = {
            str(n): layer for layer, nodes in hnsw_index.layers.items() for n in nodes
        }

        search_log.append({
            "word": actual_word,
            "hnsw": {
                "time_ms": round((end_hnsw - start_hnsw) * 1000, 2),
                "steps": int(sum(len(v) for v in hnsw_log.values())),
                "sim": float(compute_cosine_similarity(query_vector, np.atleast_2d(vectors[int(hnsw_result)]))[0])
            },
            "acorn": {
                "time_ms": round((end_acorn - start_acorn) * 1000, 2),
                "steps": int(len(acorn_path)),
                "sim": float(compute_cosine_similarity(query_vector, np.atleast_2d(vectors[int(acorn_result)]))[0])
            }
        })


        return jsonify({
            "query": actual_word,
            "query_coords": query_3d,
            "entry_point": entry_node,
            "entry_coords": pca_3d[entry_node].tolist(),

            "hnsw": {
                "result": texts[int(hnsw_result)],
                "path": {str(int(k)): [int(x) for x in v] for k, v in hnsw_log.items()},
                "time_ms": round((end_hnsw - start_hnsw) * 1000, 2),
                "num_visited": int(sum(len(v) for v in hnsw_log.values())),
                "similarity": float(compute_cosine_similarity(query_vector, np.atleast_2d(vectors[int(hnsw_result)]))[0])
            },

            "acorn": {
                "result": texts[int(acorn_result)],
                "path": [int(x) for x in acorn_path],
                "time_ms": round((end_acorn - start_acorn) * 1000, 2),
                "num_visited": int(len(acorn_path)),
                "similarity": float(compute_cosine_similarity(query_vector, np.atleast_2d(vectors[int(acorn_result)]))[0]),
                "neighbors": {
                    str(k): [int(n) for n in v]
                    for k, v in acorn_index.acorn_graph.items()
                }
            },

            "positions": {str(i): pca_3d[i].tolist() for i in range(len(vectors))},
            "layers": layer_assignments,
            "labels": {str(i): texts[i] for i in range(len(vectors))}
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/summary", methods=["GET"])
def summary():
    from random import sample
    random_words = sample(texts, 10)

    summary_data = []

    for word in random_words:
        query_vector, actual_word, idx = get_query_embedding(word, model, vectors, texts, sbert_to_glove_pca)

        # Run both searches
        h_start = time.time()
        h_res, h_log, entry_node = hnsw_index.search(query_vector)
        h_end = time.time()

        a_start = time.time()
        a_res, a_path, _ = acorn_index.search(query_vector, start_node=entry_node)
        a_end = time.time()

        summary_data.append({
            "word": actual_word,
            "hnsw": {
                "time_ms": round((h_end - h_start) * 1000, 2),
                "steps": int(sum(len(v) for v in h_log.values())),
                "sim": float(compute_cosine_similarity(query_vector, np.atleast_2d(vectors[int(h_res)]))[0])
            },
            "acorn": {
                "time_ms": round((a_end - a_start) * 1000, 2),
                "steps": int(len(a_path)),
                "sim": float(compute_cosine_similarity(query_vector, np.atleast_2d(vectors[int(a_res)]))[0]),
            }
        })

    # Combine with user-searched words
    full_report = search_log + summary_data
    return jsonify(full_report)


if __name__ == "__main__":
    app.run(debug=True, port=5050)
