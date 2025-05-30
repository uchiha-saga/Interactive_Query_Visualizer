# %%

import numpy as np
from glove_loader import load_glove_embeddings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from Interactive_Query_Visualizer.backend.graphs.hnsw_index import CompleteHNSW
from acorn1_index import ACORN1
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.io as pio

# Ensure plots open in the browser
pio.renderers.default = 'browser'

# %%

# Load GloVe embeddings
glove_path = "glove.6B.100d.txt"
texts, vectors = load_glove_embeddings(glove_path, max_words=2500)

# %%
# Build the HNSW index
hnsw_index = CompleteHNSW(vectors, M=10)

# %%
# Define the query word
query_word = "billion"

if query_word not in texts:
    print(f"⚠️ '{query_word}' not found in GloVe. Searching for closest word...")

    # Load transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query_word])[0].reshape(1, -1)

    # Reduce dimension from 384 → 100 to match GloVe
    pca = PCA(n_components=100)
    pca.fit(model.encode(texts[:1000]))  # You can adjust number of texts here for speed
    reduced_query = pca.transform(query_embedding)[0]
    reduced_query /= np.linalg.norm(reduced_query)

    # Find closest GloVe word
    sims = cosine_similarity([reduced_query], vectors)[0]
    best_index = np.argmax(sims)
    fallback_word = texts[best_index]
    query_vector = vectors[best_index]

    print(f"🔁 Using closest GloVe word instead: '{fallback_word}' (similarity = {sims[best_index]:.4f})")
    query_word = fallback_word
else:
    query_vector = vectors[texts.index(query_word)]

# Run HNSW search
hnsw_nearest, hnsw_log, entry_node = hnsw_index.search(query_vector)

# Run ACORN-1 search
acorn_index = ACORN1(hnsw_index)
acorn_nearest, acorn_path, acorn_entry = acorn_index.search(query_vector)

# Fit PCA on embeddings for 3D visualization
pca = PCA(n_components=3)
reduced_embeddings = pca.fit_transform(vectors)
query_reduced = pca.transform(query_vector.reshape(1, -1))[0]
entry_coords = reduced_embeddings[entry_node]
acorn_coords = reduced_embeddings[acorn_entry]

# Layer assignment for coloring
layers_assignment = [0] * len(vectors)
for layer, nodes in hnsw_index.layers.items():
    for node in nodes:
        layers_assignment[node] = layer

# Color mapping
layer_colors = {0: "gray", 1: "orange", 2: "gold"}
traces = []

# Layer-by-layer points
for layer_num in [0, 1, 2]:
    indices = [i for i in range(len(vectors)) if layers_assignment[i] == layer_num]
    coords = reduced_embeddings[indices]
    traces.append(go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode='markers',
        marker=dict(size=3, color=layer_colors[layer_num]),
        name=f"Layer {layer_num}",
        text=[texts[i] for i in indices],
        hoverinfo='text'
    ))

# Query point
traces.append(go.Scatter3d(
    x=[query_reduced[0]], y=[query_reduced[1]], z=[query_reduced[2]],
    mode='markers',
    marker=dict(size=8, color='red', symbol='x'),
    name='Query Word',
    text=[query_word],
    hoverinfo='text'
))

# HNSW Entry point
traces.append(go.Scatter3d(
    x=[entry_coords[0]], y=[entry_coords[1]], z=[entry_coords[2]],
    mode='markers',
    marker=dict(size=8, color='blue', symbol='diamond'),
    name='HNSW Entry',
    text=[texts[entry_node]],
    hoverinfo='text'
))

# ACORN Entry point (same as HNSW)
traces.append(go.Scatter3d(
    x=[acorn_coords[0]], y=[acorn_coords[1]], z=[acorn_coords[2]],
    mode='markers',
    marker=dict(size=8, color='purple', symbol='circle'),
    name='ACORN Entry',
    text=[texts[acorn_entry]],
    hoverinfo='text'
))

# HNSW Traversal Paths
for layer, nodes in hnsw_log.items():
    coords = reduced_embeddings[nodes]
    traces.append(go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode='lines+markers',
        line=dict(color='black', width=3, dash='dash'),
        marker=dict(size=5, color='black'),
        name=f"HNSW Path (Layer {layer})",
        text=[texts[i] for i in nodes],
        hoverinfo='text'
    ))

# ACORN Traversal Path
acorn_coords_path = reduced_embeddings[acorn_path]
traces.append(go.Scatter3d(
    x=acorn_coords_path[:, 0], y=acorn_coords_path[:, 1], z=acorn_coords_path[:, 2],
    mode='lines+markers',
    line=dict(color='green', width=3),
    marker=dict(size=5, color='green'),
    name='ACORN Path',
    text=[texts[i] for i in acorn_path],
    hoverinfo='text'
))

# Layout and render
layout = go.Layout(
    title=f"3D Traversal Comparison for Query: '{query_word}'",
    scene=dict(
        xaxis_title="PCA Dimension 1",
        yaxis_title="PCA Dimension 2",
        zaxis_title="PCA Dimension 3"
    ),
    margin=dict(l=0, r=0, b=0, t=30)
)

fig = go.Figure(data=traces, layout=layout)
fig.show()

# %%
# bunch of 10 random words
# check the traversal time and number of nodes
# how near was the nearest neighot (result) 
# %%
