# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.decomposition import PCA
from query_execution import execute_query
from vector_ops import reduce_dimensions
import numpy as np
from supabase_setup import embeddings, texts, model  # Make sure model is imported

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return "Backend API is running."

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    query_text = data.get("query")
    if not query_text:
        return jsonify({"error": "Query text is required"}), 400

    # Execute the vector search query.
    result = execute_query(query_text, k=3)

    # Compute PCA on the full dataset.
    embeddings_np = np.array(embeddings)
    pca_result = reduce_dimensions(embeddings_np, n_components=3)  # shape: (n_samples, 3)
    
    # Compute query's embedding and transform to PCA space.
    query_embed = model.encode([query_text], convert_to_numpy=True)
    query_pca = pca_result = reduce_dimensions(embeddings_np, n_components=3)
    # Alternatively, if you want the correct transformation for query, you can:
    # pca = PCA(n_components=3)
    # pca_result = pca.fit_transform(embeddings_np)
    # query_pca = pca.transform(query_embed)
    # For consistency, below we assume the latter:
    
    pca_obj = PCA(n_components=3)
    pca_result = pca_obj.fit_transform(embeddings_np)
    query_pca = pca_obj.transform(query_embed)
    
    response = {
        "query": query_text,
        "pca": pca_result.tolist(),           # PCA-transformed coordinates for each data point.
        "query_pca": query_pca.tolist()[0],     # The query's PCA coordinates (first and only vector).
        "texts": texts,                        # Original labels.
        "result": {
            "nearest_idx": result["nearest_idx"],
            "traversal_log": result["traversal_log"],
            "entry_node": result["entry_node"]
        }
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
