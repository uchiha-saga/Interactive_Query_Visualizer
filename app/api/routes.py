from flask import Blueprint, jsonify, request
import numpy as np
from sklearn.decomposition import PCA
import json
import traceback  # Add import for error tracking
from app.database.db_manager import get_db_connection, get_embeddings_from_db
from app.utils.vector_operations import reduce_dimensions, compute_cosine_similarity, compute_euclidean_distance

api = Blueprint('api', __name__, url_prefix='/api')

@api.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

@api.route('/embeddings', methods=['GET'])
def get_embeddings():
    try:
        # Get embeddings from database or sample data
        embeddings, metadata = get_embeddings_from_db()
        
        # Generate sample data if none was returned
        if embeddings is None or len(embeddings) == 0:
            # Create random embeddings for demonstration
            num_samples = 100
            dimension = 384
            embeddings = np.random.rand(num_samples, dimension)
            metadata = [{"id": i, "label": f"Sample {i}"} for i in range(num_samples)]
        
        # Reduce dimensions for visualization
        reduced_data = reduce_dimensions(embeddings)
        
        # Combine with metadata
        result = []
        for i, point in enumerate(reduced_data):
            result.append({
                "position": point.tolist(),
                "metadata": metadata[i] if i < len(metadata) else {}
            })
            
        return jsonify({"data": result, "status": "success"})
    except Exception as e:
        print(f"Error in /embeddings endpoint: {str(e)}")
        print(traceback.format_exc())  # Print the full traceback
        return jsonify({"error": str(e), "status": "error"}), 500

@api.route('/query', methods=['POST'])
def execute_query():
    try:
        data = request.json
        query_vector = np.array(data.get('query_vector', []), dtype=np.float64)
        k = data.get('k', 10)
        
        if len(query_vector) == 0:
            return jsonify({"error": "Query vector is required", "status": "error"}), 400
            
        # Get embeddings from database or sample data
        embeddings, metadata = get_embeddings_from_db()
        if embeddings is None or len(embeddings) == 0:
            return jsonify({"error": "No embeddings available", "status": "error"}), 404
            
        # Compute similarities using cosine similarity
        similarities = compute_cosine_similarity(query_vector, embeddings)
        
        # Get top k results
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Prepare results
        results = []
        for idx in top_k_indices:
            results.append({
                "id": int(idx),
                "distance": float(1 - similarities[idx]),  # Convert similarity to distance
                "metadata": metadata[idx] if idx < len(metadata) else {}
            })
        
        return jsonify({
            "results": results,
            "status": "success"
        })
    except Exception as e:
        print(f"Error in /query endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e), "status": "error"}), 500 