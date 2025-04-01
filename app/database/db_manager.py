import os
import psycopg2
import numpy as np
import traceback
from dotenv import load_dotenv
import sqlite3
from typing import Tuple, List, Dict, Optional
import json

# Load environment variables
load_dotenv()

def get_db_connection():
    """Get a connection to the SQLite database."""
    return sqlite3.connect('embeddings.db')

def init_db():
    """Initialize the database with required tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create embeddings table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        vector BLOB NOT NULL,
        metadata TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()

def get_embeddings_from_db() -> Tuple[Optional[np.ndarray], List[Dict]]:
    """
    Retrieve embeddings and metadata from the database.
    
    Returns:
        Tuple[Optional[np.ndarray], List[Dict]]: Tuple containing:
            - numpy array of embeddings (or None if no data)
            - list of metadata dictionaries
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get all embeddings and metadata
        cursor.execute('SELECT id, vector, metadata FROM embeddings')
        rows = cursor.fetchall()
        
        if not rows:
            return None, []
            
        # Convert binary vectors to numpy arrays
        vectors = []
        metadata = []
        for row in rows:
            id_, vector_blob, metadata_json = row
            vector = np.frombuffer(vector_blob, dtype=np.float64)
            vectors.append(vector)
            metadata_dict = json.loads(metadata_json) if metadata_json else {}
            metadata_dict['id'] = id_
            metadata.append(metadata_dict)
            
        return np.array(vectors), metadata
        
    except Exception as e:
        print(f"Error retrieving embeddings: {str(e)}")
        return None, []
    finally:
        conn.close()

def insert_embedding(vector: np.ndarray, metadata: Dict = None) -> bool:
    """
    Insert a new embedding into the database.
    
    Args:
        vector (np.ndarray): The embedding vector
        metadata (Dict, optional): Additional metadata to store
        
    Returns:
        bool: True if successful, False otherwise
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Convert vector to binary
        vector_blob = vector.tobytes()
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute(
            'INSERT INTO embeddings (vector, metadata) VALUES (?, ?)',
            (vector_blob, metadata_json)
        )
        conn.commit()
        return True
        
    except Exception as e:
        print(f"Error inserting embedding: {str(e)}")
        return False
    finally:
        conn.close()

def delete_embedding(id_: int) -> bool:
    """
    Delete an embedding from the database.
    
    Args:
        id_ (int): The ID of the embedding to delete
        
    Returns:
        bool: True if successful, False otherwise
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('DELETE FROM embeddings WHERE id = ?', (id_,))
        conn.commit()
        return True
        
    except Exception as e:
        print(f"Error deleting embedding: {str(e)}")
        return False
    finally:
        conn.close()

def setup_database():
    """Setup database tables for vector storage"""
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            print("Could not connect to database. Skipping database setup.")
            return
            
        with conn.cursor() as cur:
            # Create embeddings table if not exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY,
                    embedding vector(384),
                    text_data TEXT,
                    metadata JSONB
                );
            """)
            
            # Create index for fast similarity search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS embeddings_vector_idx 
                ON embeddings 
                USING ivfflat (embedding vector_l2_ops)
                WITH (lists = 100);
            """)
            
            conn.commit()
            print("Database setup complete")
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Database setup error: {error}")
        print(traceback.format_exc())
    finally:
        if conn is not None:
            conn.close()

def generate_sample_embeddings(num_samples: int = 100, dimension: int = 384) -> Tuple[np.ndarray, List[Dict]]:
    """
    Generate sample embeddings for testing.
    
    Args:
        num_samples (int): Number of sample embeddings to generate
        dimension (int): Dimension of each embedding vector
        
    Returns:
        Tuple[np.ndarray, List[Dict]]: Tuple containing:
            - numpy array of random embeddings
            - list of sample metadata dictionaries
    """
    embeddings = np.random.rand(num_samples, dimension)
    metadata = [{"id": i, "label": f"Sample {i}"} for i in range(num_samples)]
    return embeddings, metadata 