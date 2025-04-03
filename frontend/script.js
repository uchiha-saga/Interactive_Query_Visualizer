document.getElementById('search-button').addEventListener('click', () => {
  const queryText = document.getElementById('query-input').value;
  if (!queryText) return;

  fetch('http://127.0.0.1:5000/query', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ query: queryText })
  })
  .then(response => response.json())
  .then(data => {
    renderPlots(data);
  })
  .catch(error => {
    console.error("Error:", error);
  });
});

function renderPlots(data) {
  const pcaCoords = data.pca;       // [[x, y, z], ...]
  const queryPCA = data.query_pca;  // [x, y, z]
  const texts = data.texts;
  const nearestIdx = data.result.nearest_idx;
  const traversalLog = data.result.traversal_log;
  const entryNode = data.result.entry_node;
  const queryText = data.query;

  // =========================
  // Left Plot: PCA of Embeddings
  // =========================
  const traceEmbeddings = {
    x: pcaCoords.map(d => d[0]),
    y: pcaCoords.map(d => d[1]),
    z: pcaCoords.map(d => d[2]),
    mode: 'markers',
    type: 'scatter3d',
    name: 'Data Embeddings',
    marker: {
      size: 5,
      color: 'gray'
    },
    text: texts,
    hoverlabel: {
      bgcolor: 'lightgrey'
    }
  };

  const traceQuery = {
    x: [queryPCA[0]],
    y: [queryPCA[1]],
    z: [queryPCA[2]],
    mode: 'markers',
    type: 'scatter3d',
    name: 'Query',
    marker: {
      size: 8,
      color: 'red',
      symbol: 'x'
    },
    hoverlabel: {
      bgcolor: 'lightgrey'
    }
  };

  const layoutPCA = {
    title: '3D PCA of Embeddings',
    scene: {
      xaxis: { title: 'PC1' },
      yaxis: { title: 'PC2' },
      zaxis: { title: 'PC3' }
    }
  };

  Plotly.newPlot('pca-plot', [traceEmbeddings, traceQuery], layoutPCA);

  // =========================
  // Right Plot: Traversal Path
  // =========================
  const traceData = {
    x: pcaCoords.map(d => d[0]),
    y: pcaCoords.map(d => d[1]),
    z: pcaCoords.map(d => d[2]),
    mode: 'markers',
    type: 'scatter3d',
    marker: { size: 5, color: 'lightgray' },
    name: 'Data Embeddings',
    text: texts,
    hoverlabel: {
      bgcolor: 'lightgrey'
    }
  };

  const traceQuery2 = {
    x: [queryPCA[0]],
    y: [queryPCA[1]],
    z: [queryPCA[2]],
    mode: 'markers',
    type: 'scatter3d',
    name: 'Query',
    marker: { size: 8, color: 'red', symbol: 'x' },
    hoverlabel: {
      bgcolor: 'lightgrey'
    }
  };

  const pathIndices = [entryNode].concat(traversalLog);
  const pathX = pathIndices.map(i => pcaCoords[i][0]);
  const pathY = pathIndices.map(i => pcaCoords[i][1]);
  const pathZ = pathIndices.map(i => pcaCoords[i][2]);
  const pathLabels = pathIndices.map(i => texts[i]);

  const tracePath = {
    x: pathX,
    y: pathY,
    z: pathZ,
    mode: 'lines+markers',
    type: 'scatter3d',
    line: { color: 'blue', dash: 'dash' },
    marker: { size: 5, color: 'blue' },
    name: 'Traversal Path',
    text: pathLabels,
    hoverlabel: {
      bgcolor: 'lightgrey'
    }
  };

  const nnPoint = pcaCoords[nearestIdx];
  const traceNearest = {
    x: [nnPoint[0]],
    y: [nnPoint[1]],
    z: [nnPoint[2]],
    mode: 'markers',
    type: 'scatter3d',
    marker: { size: 8, color: 'green', symbol: 'circle' },
    name: 'Nearest Neighbor',
    text: [`Nearest: ${texts[nearestIdx]}`],
    hoverlabel: {
      bgcolor: 'lightgrey'
    }
  };

  const enPoint = pcaCoords[entryNode];
  const traceEntry = {
    x: [enPoint[0]],
    y: [enPoint[1]],
    z: [enPoint[2]],
    mode: 'markers',
    type: 'scatter3d',
    marker: { size: 10, color: 'blue', symbol: 'diamond' },
    name: 'Entry Node',
    text: [`Entry: ${texts[entryNode]}`],
    hoverlabel: {
      bgcolor: 'lightgrey'
    }
  };

  const layoutTraversal = {
    title: '3D Traversal Path in PCA Space',
    scene: {
      xaxis: { title: 'PC1' },
      yaxis: { title: 'PC2' },
      zaxis: { title: 'PC3' }
    }
  };

  Plotly.newPlot('traversal-plot', [traceData, traceQuery2, tracePath, traceNearest, traceEntry], layoutTraversal);

  // =========================
  // Result Text Box
  // =========================
  document.getElementById('result-text').innerHTML =
    `<strong>Search result for '${queryText}':</strong><br>
     <span class="result-item"><strong>Nearest Neighbor:</strong> ${texts[nearestIdx]}</span><br>
     <span class="result-item"><strong>Entry Node:</strong> ${texts[entryNode]}</span>`;
}
