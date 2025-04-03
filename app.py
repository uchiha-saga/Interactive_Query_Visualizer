# app.py
import dash
from dash import html, dcc, Input, Output, State
import plotly.graph_objs as go
import numpy as np
import dash_bootstrap_components as dbc
from supabase_setup import model, embeddings, texts
from query_execution import execute_query
from sklearn.decomposition import PCA

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dbc.Container([
        dbc.Row([dbc.Col(html.H1("Vector DB 3D Visualization UI"), width=12)]),
        dbc.Row([
            dbc.Col(
                dcc.Input(
                    id='query-input', type='text', placeholder='Enter your search query',
                    style={'width': '100%'}
                ), width=8
            ),
            dbc.Col(
                dbc.Button("Search", id='search-button', n_clicks=0), width=4
            )
        ], style={'padding': '20px'}),
        dbc.Row([
            dbc.Col(dcc.Graph(id='pca-graph'), width=6),
            dbc.Col(dcc.Graph(id='traversal-graph'), width=6)
        ]),
        dbc.Row([
            dbc.Col(html.Div(id='result-output', style={'padding': '20px', 'fontSize': '20px'}), width=12)
        ])
    ])
])

@app.callback(
    [Output('pca-graph', 'figure'),
     Output('traversal-graph', 'figure'),
     Output('result-output', 'children')],
    [Input('search-button', 'n_clicks')],
    [State('query-input', 'value')]
)
def update_graphs(n_clicks, query_text):
    n_components = 3
    # Compute PCA on the full dataset
    embeddings_np = np.array(embeddings)
    pca = PCA(n_components=n_components)
    dataset_pca = pca.fit_transform(embeddings_np)
    
    # Create PCA 3D plot of the dataset
    pca_fig = go.Figure(data=go.Scatter3d(
        x=dataset_pca[:, 0],
        y=dataset_pca[:, 1],
        z=dataset_pca[:, 2],
        mode='markers',
        marker=dict(size=5),
        text=texts,
        name='Data Embeddings'
    ))
    pca_fig.update_layout(
        title="3D PCA of Embeddings",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3")
    )
    
    # If no query is provided, return the PCA figure, an empty traversal graph, and no result text.
    if not query_text:
        traversal_fig = go.Figure()
        traversal_fig.update_layout(title="Traversal Path")
        return pca_fig, traversal_fig, ""
    
    # Execute the query using your custom HNSW search
    result = execute_query(query_text, k=3)
    # result now returns keys: "nearest_idx", "traversal_log", and "entry_node"
    nearest_idx = result["nearest_idx"]
    traversal_log = result["traversal_log"]  # List of nodes where improvement occurred.
    entry_idx = result["entry_node"]
    
    # Get query embedding and transform with the PCA model
    query_embed = model.encode([query_text], convert_to_numpy=True)
    query_pca = pca.transform(query_embed)
    
    # Add query point to the PCA graph (red cross)
    pca_fig.add_trace(go.Scatter3d(
        x=[query_pca[0, 0]],
        y=[query_pca[0, 1]],
        z=[query_pca[0, 2]],
        mode='markers',
        marker=dict(color='red', size=8, symbol='x'),
        name='Query'
    ))
    
    # Create the traversal graph.
    traversal_fig = go.Figure()
    
    # Add the query marker (red cross) separately; it is not connected by a line.
    traversal_fig.add_trace(go.Scatter3d(
         x=[query_pca[0, 0]],
         y=[query_pca[0, 1]],
         z=[query_pca[0, 2]],
         mode='markers',
         marker=dict(color='red', size=8, symbol='x'),
         name='Query'
    ))
    
    # Build the traversal path: start at the entry node, then follow any improvements.
    if traversal_log and len(traversal_log) > 0:
        # The path starts at the entry node and then follows the improvement nodes.
        path_indices = [entry_idx] + traversal_log
    else:
        # If no improvement occurred, the path only contains the entry node (which equals nearest).
        path_indices = [entry_idx]
    
    # Extract PCA coordinates for the traversal path.
    path_coords = np.array([dataset_pca[i] for i in path_indices])
    path_labels = [texts[i] for i in path_indices]
    
    # Draw the traversal path as a blue line connecting the nodes.
    if len(path_coords) > 1:
        traversal_fig.add_trace(go.Scatter3d(
            x=path_coords[:, 0],
            y=path_coords[:, 1],
            z=path_coords[:, 2],
            mode='lines+markers',
            marker=dict(size=5, color='blue'),
            line=dict(color='blue', dash='dash'),
            text=path_labels,
            name='Traversal Path'
        ))
    else:
        # Only one node in the path; show it as a marker.
        traversal_fig.add_trace(go.Scatter3d(
            x=[path_coords[0, 0]],
            y=[path_coords[0, 1]],
            z=[path_coords[0, 2]],
            mode='markers',
            marker=dict(size=5, color='blue'),
            text=path_labels,
            name='Traversal Path'
        ))
    
    # Add a distinct marker for the nearest neighbor (green)
    nearest_point = dataset_pca[nearest_idx]
    traversal_fig.add_trace(go.Scatter3d(
         x=[nearest_point[0]],
         y=[nearest_point[1]],
         z=[nearest_point[2]],
         mode='markers',
         marker=dict(color='green', size=10, symbol='circle'),
         text=[f"Nearest Neighbor: {texts[nearest_idx]}"],
         name='Nearest Neighbor'
    ))
    
    # Optionally add a marker for the entry node (if not already in the path)
    entry_point = dataset_pca[entry_idx]
    traversal_fig.add_trace(go.Scatter3d(
         x=[entry_point[0]],
         y=[entry_point[1]],
         z=[entry_point[2]],
         mode='markers',
         marker=dict(color='blue', size=12, symbol='diamond'),
         text=[f"Entry Node: {texts[entry_idx]}"],
         name='Entry Node'
    ))

    # (Here it will already be the first element of the traversal path, so we don't add a duplicate marker.)
    
    traversal_fig.update_layout(
         title="3D Traversal Path in PCA Space",
         scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3")
    )
    
    # Prepare the result text.
    result_text = f"Search result for '{query_text}': Nearest Neighbor -> {texts[nearest_idx]}, Entry Node -> {texts[entry_idx]}"
    
    return pca_fig, traversal_fig, result_text


if __name__ == '__main__':
    app.run(debug=True)
