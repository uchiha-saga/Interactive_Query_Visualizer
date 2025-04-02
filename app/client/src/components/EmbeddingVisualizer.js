import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios';

const EmbeddingVisualizer = () => {
  const [embeddingData, setEmbeddingData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [queryVector, setQueryVector] = useState(null);
  const [queryResults, setQueryResults] = useState(null);
  const [isSearching, setIsSearching] = useState(false);

  useEffect(() => {
    fetchEmbeddings();
  }, []);

  const fetchEmbeddings = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/embeddings');
      if (response.data.status === 'success') {
        setEmbeddingData(response.data.data);
      } else {
        setError('Failed to load embeddings data');
      }
    } catch (err) {
      setError('Error fetching embeddings: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleQuery = async () => {
    if (!queryVector) return;
    
    try {
      setIsSearching(true);
      const response = await axios.post('/api/query', {
        query_vector: queryVector,
        k: 10
      });
      
      if (response.data.status === 'success') {
        setQueryResults(response.data.results);
      }
    } catch (err) {
      console.error('Query error:', err);
    } finally {
      setIsSearching(false);
    }
  };

  const prepareData = () => {
    if (!embeddingData || embeddingData.length === 0) {
      return null;
    }

    // Extract x, y, z coordinates
    const x = embeddingData.map(point => point.position[0]);
    const y = embeddingData.map(point => point.position[1]);
    const z = embeddingData.map(point => point.position[2]);
    
    // Create hover text with metadata
    const text = embeddingData.map(point => {
      const metadata = point.metadata || {};
      return Object.entries(metadata)
        .map(([key, value]) => `${key}: ${value}`)
        .join('<br>');
    });

    // Create marker colors based on query results
    const colors = embeddingData.map((_, index) => {
      if (queryResults) {
        const isResult = queryResults.some(result => result.id === index);
        return isResult ? 'rgb(255, 0, 0)' : 'rgb(23, 190, 207)';
      }
      return 'rgb(23, 190, 207)';
    });

    return [{
      type: 'scatter3d',
      mode: 'markers',
      x: x,
      y: y,
      z: z,
      text: text,
      hoverinfo: 'text',
      marker: {
        size: 5,
        color: colors,
        opacity: 0.8
      }
    }];
  };

  const handlePointClick = (data) => {
    const pointIndex = data.points[0].pointIndex;
    if (pointIndex >= 0 && pointIndex < embeddingData.length) {
      setSelectedPoint(embeddingData[pointIndex]);
    }
  };

  if (loading) {
    return <div>Loading embeddings...</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  const plotData = prepareData();
  if (!plotData) {
    return <div>No embedding data available</div>;
  }

  return (
    <div className="embedding-visualizer">
      <h2>Interactive Embedding Explorer</h2>
      
      <div className="query-section">
        <h3>Query Search</h3>
        <div className="query-input">
          <input
            type="text"
            placeholder="Enter query vector (comma-separated numbers)"
            onChange={(e) => {
              const vector = e.target.value.split(',').map(num => parseFloat(num.trim()));
              if (vector.every(num => !isNaN(num))) {
                setQueryVector(vector);
              }
            }}
          />
          <button 
            onClick={handleQuery}
            disabled={!queryVector || isSearching}
          >
            {isSearching ? 'Searching...' : 'Search'}
          </button>
        </div>
      </div>

      <div className="visualization-container">
        <Plot
          data={plotData}
          layout={{
            width: 800,
            height: 600,
            title: '3D Embedding Visualization',
            scene: {
              xaxis: { title: 'X' },
              yaxis: { title: 'Y' },
              zaxis: { title: 'Z' }
            },
            margin: {
              l: 0,
              r: 0,
              b: 0,
              t: 50
            }
          }}
          onClick={handlePointClick}
          config={{
            displayModeBar: true,
            responsive: true
          }}
        />
      </div>
      
      {selectedPoint && (
        <div className="point-details">
          <h3>Selected Point Details</h3>
          <div>
            <strong>Position:</strong> ({selectedPoint.position.join(', ')})
          </div>
          <div>
            <strong>Metadata:</strong>
            <pre>{JSON.stringify(selectedPoint.metadata, null, 2)}</pre>
          </div>
        </div>
      )}

      {queryResults && (
        <div className="query-results">
          <h3>Query Results</h3>
          <div className="results-list">
            {queryResults.map((result, index) => (
              <div key={index} className="result-item">
                <span>ID: {result.id}</span>
                <span>Distance: {result.distance.toFixed(4)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default EmbeddingVisualizer; 