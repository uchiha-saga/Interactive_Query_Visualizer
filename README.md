# Interactive Embedding Explorer and Query Execution Visualizer

A full-stack application for visualizing and exploring embeddings in vector databases. This tool provides an interactive 3D visualization of embeddings using dimensionality reduction techniques and helps understand PostgreSQL vector query execution.

## Features

- 3D embedding visualization with PCA dimensionality reduction
- Interactive exploration of embedding relationships
- Basic query execution visualization
- Real-time query search visualization

## Prerequisites

- Python 3.8+
- PostgreSQL with pgvector extension
- Node.js 14+
- npm or yarn

## Setup

### Backend Setup

1. Create a Python virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure PostgreSQL:
   - Make sure PostgreSQL is installed and running
   - Install the pgvector extension for PostgreSQL
   - Update the `.env` file with your database credentials

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

## Running the Application

### Start the Backend

1. From the project root, activate the virtual environment if not already activated
2. Run the Flask application:
   ```
   python app.py
   ```
   The backend will be available at http://localhost:5000

### Start the Frontend

1. In a separate terminal, navigate to the fronted directory:
   ```
   cd front
   ```

2. Run the React development server:
   ```
   python -m http.server 8000
   ```
   The frontend will be available at http://localhost:8000

## Development

This project implements the following key components:

1. **Embedding Explorer**: Visualizes embeddings in an interactive 3D space
2. **Query Execution Flow**: Illustrates how PostgreSQL processes vector similarity queries
3. **Query Search Visualization**: Provides real-time query visualization and HNSW traversal

## Project Structure

```
.
├── backend/
│   ├── app.py               # Used to start the backend (activates the python files)
│   ├── hnsw_index.py        # Creates our customised HNSW with Euclidean distance
│   ├── query_execution.py   # Embeds the search query and displays in the 3D plot
│   ├── supabase_setup.py    # Connects to the Supabase, initializes the embedding model
                              and inserts the new embedding in the database
│   ├── vector_ops.py        # Contains the PCA implementation and its helper functions
│   ├── visualize.py         # Reduces the vector dimensions to 3D PCA for plotting
├── frontend/
│   ├── index.html           # Displays the interface and the plots
│   ├── script.js            # Connects front-end and back-end
│   ├── styles.css           # Customizes the front-end User interface
└── requirements.txt         # Python dependencies
```

## HNSW Node Traversal

The code visualizes the traversal of node we search query is entered and displays the nearest neighboring node to the search query.

![Screenshot 2025-04-03 at 11 09 00 AM](https://github.com/user-attachments/assets/b7d09437-5a5f-4228-a8c5-da6e5f28bfe1)

![Screenshot 2025-04-03 at 11 09 36 AM](https://github.com/user-attachments/assets/6d2aa4fd-40be-4648-acab-d0a8808baa81)

## Authors

- Gaurang Kamat
- Sunho Park
