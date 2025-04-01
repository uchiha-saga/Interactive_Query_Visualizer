import React from 'react';
import './App.css';
import EmbeddingVisualizer from './components/EmbeddingVisualizer';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Interactive Embedding Explorer and Query Execution Visualizer</h1>
      </header>
      <main>
        <EmbeddingVisualizer />
      </main>
      <footer>
        <p>Created by Gaurang Kamat and Sunho Park</p>
      </footer>
    </div>
  );
}

export default App; 