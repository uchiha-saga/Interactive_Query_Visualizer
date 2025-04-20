document.getElementById("search-btn").onclick = async () => {
    const word = document.getElementById("word-input").value;

    try {
        const res = await fetch("/query", {
            method: "POST",
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ word })
        });

        const data = await res.json();
        if (data.error) throw new Error(data.error);

        const labels = data.labels;
        const coords = data.positions;
        const layers = data.layers;

        const queryCoords = data.query_coords;
        const entryCoords = data.entry_coords;

        const layerColors = { 0: 'gray', 1: 'orange', 2: 'gold' };

        // 📊 PLOT 1: Layer Visualization (true PCA positions, all layer nodes, no Z-offset)
        const layerNameMap = { 0: 'Base Layer', 1: 'Middle Layer', 2: 'Entry Layer' };

        // Organize all nodes by layer
        const allLayerPoints = {};
        for (let node in coords) {
            const rawLayer = layers[node];
            if (rawLayer === undefined) continue;
            if (!allLayerPoints[rawLayer]) allLayerPoints[rawLayer] = [];
            allLayerPoints[rawLayer].push({
                id: node,
                pos: coords[node],
                label: labels[node]
            });
        }

        const layerTracesPlot1 = Object.entries(allLayerPoints).map(([layer, nodes]) => ({
            type: 'scatter3d',
            mode: 'markers',
            name: layerNameMap[layer],
            x: nodes.map(n => n.pos[0]),
            y: nodes.map(n => n.pos[1]),
            z: nodes.map(n => n.pos[2]),
            text: nodes.map(n => n.label),
            hoverinfo: 'text',
            marker: {
                size: 3,
                color: layerColors[layer] || 'gray'
            }
        }));

        // Query marker (in real position)
        const queryTracePlot1 = {
            type: 'scatter3d',
            mode: 'markers',
            name: 'Query Word',
            x: [queryCoords[0]],
            y: [queryCoords[1]],
            z: [queryCoords[2]],
            text: [data.query],
            hoverinfo: 'text',
            marker: {
                size: 8,
                color: 'red',
                symbol: 'x'
            }
        };

        Plotly.newPlot("layer-plot", [...layerTracesPlot1, queryTracePlot1], {
            title: {
                text: 'HNSW Structure by Layer',
                font: { family: 'Inter, sans-serif', size: 18 },
                pad: { t: 40, b: 10 }
            },
            scene: {
                xaxis: { title: 'PCA 1' },
                yaxis: { title: 'PCA 2' },
                zaxis: { title: 'PCA 3' }
            },
            margin: { l: 0, r: 0, b: 0, t: 40 },
            showlegend: true
        });
        


        // 📊 PLOT 2: Traversal Visualization
        const layerTracesPlot2 = Object.entries(allLayerPoints).map(([layer, nodes]) => ({
            type: 'scatter3d',
            mode: 'markers',
            name: {
                "0": "Base Layer",
                "1": "Middle Layer",
                "2": "Entry Layer"
              }[layer] || `Layer ${layer}`,              
            x: nodes.map(n => n.pos[0]),
            y: nodes.map(n => n.pos[1]),
            z: nodes.map(n => n.pos[2]),
            text: nodes.map(n => n.label),
            hoverinfo: 'text',
            marker: {
                size: 3,
                color: layerColors[layer] || 'gray',
                opacity: 0.4
            }
        }));

        const queryTrace = {
            type: 'scatter3d',
            mode: 'markers',
            name: 'Query Word',
            x: [queryCoords[0]],
            y: [queryCoords[1]],
            z: [queryCoords[2]],
            text: [data.query],
            hoverinfo: 'text',
            marker: {
                size: 8,
                color: 'red',
                symbol: 'x'
            }
        };

        const entryTrace = {
            type: 'scatter3d',
            mode: 'markers',
            name: 'Entry Point',
            x: [entryCoords[0]],
            y: [entryCoords[1]],
            z: [entryCoords[2]],
            text: ['Entry'],
            hoverinfo: 'text',
            marker: {
                size: 8,
                color: 'blue',
                symbol: 'diamond'
            }
        };

        const hnswLayerTraces = Object.entries(data.hnsw.path).map(([layer, nodeList]) => {
            const pathCoords = nodeList.map(n => coords[n.toString()]);
            return {
                type: 'scatter3d',
                mode: 'lines+markers',
                name: `HNSW Path (${{
                    "0": "Base Layer",
                    "1": "Middle Layer",
                    "2": "Entry Layer"
                  }[layer] || `Layer ${layer}`})`,                  
                x: pathCoords.map(c => c[0]),
                y: pathCoords.map(c => c[1]),
                z: pathCoords.map(c => c[2]),
                text: nodeList.map(n => labels[n.toString()]),
                hoverinfo: 'text',
                marker: { size: 5, color: 'black' },
                line: { width: 3, color: 'black', dash: 'dash' }
            };
        });

        const acornCoords = data.acorn.path.map(i => coords[i.toString()]);
        const acornTrace = {
            type: 'scatter3d',
            mode: 'lines+markers',
            name: 'ACORN-1 Path',
            x: acornCoords.map(c => c[0]),
            y: acornCoords.map(c => c[1]),
            z: acornCoords.map(c => c[2]),
            text: data.acorn.path.map(i => labels[i.toString()]),
            hoverinfo: 'text',
            marker: { size: 5, color: 'green' },
            line: { width: 3, color: 'green' }
        };

        const acornRadiusTraces = [];

        data.acorn.path.forEach(nodeIdx => {
        const neighbors = data.acorn.neighbors[nodeIdx.toString()] || [];
        const positions = neighbors.map(i => coords[i.toString()]);
        const labelsList = neighbors.map(i => labels[i.toString()]);

        acornRadiusTraces.push({
            type: 'scatter3d',
            mode: 'markers',
            name: `ACORN Radius Neighbors - ${labels[nodeIdx.toString()]}`,
            x: positions.map(p => p[0]),
            y: positions.map(p => p[1]),
            z: positions.map(p => p[2]),
            text: labelsList,
            hoverinfo: 'text',
            marker: {
            size: 3,
            color: 'rgba(0, 255, 0, 0.2)', // translucent green
            symbol: 'circle'
            },
            visible: 'legendonly'
        });
        });


        Plotly.newPlot("traversal-plot", [
            ...layerTracesPlot2,
            queryTrace,
            entryTrace,
            ...hnswLayerTraces,
            acornTrace,
            ...acornRadiusTraces
        ], {
            title: {
                text: `Search Traversal for '${word}'`,
                font: { family: 'Inter, sans-serif', size: 18 },
                pad: { t: 40, b: 10 }
            },
            scene: {
                xaxis: { title: 'PCA 1' },
                yaxis: { title: 'PCA 2' },
                zaxis: { title: 'PCA 3' }
            },
            margin: { l: 0, r: 0, b: 0, t: 40 },
            showlegend: true
        });
        

        //  Report panel
        document.getElementById("report").innerHTML = `
            <strong>Your Query:</strong> ${word}<br>

            <strong>Closest Match in GloVe:</strong> ${data.query}<br><br>

            <strong>HNSW Result:</strong> ${data.hnsw.result}<br>
            Time: ${data.hnsw.time_ms}ms 
            <br>
            Steps: ${data.hnsw.num_visited}
            <br>
            Cosine Similarity: ${data.hnsw.similarity.toFixed(4)}
            <br><br>

            <strong>ACORN-1 Result:</strong> ${data.acorn.result}<br>
            Time: ${data.acorn.time_ms}ms  
            <br>
            Steps: ${data.acorn.num_visited}
            <br>
            Cosine Similarity: ${data.acorn.similarity.toFixed(4)}
            <br>
        `;

        // 🧠 Tab Switching Logic
        document.querySelectorAll('.tab-link').forEach(link => {
            link.addEventListener('click', (e) => {
            e.preventDefault();
        
            const tab = link.getAttribute('data-tab');
        
            document.querySelectorAll('.tab-link').forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        
            document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));
            document.getElementById(`${tab}-tab`).classList.add('active');
            });
        });
          


    } catch (err) {
        console.error("Error occurred:", err);
        document.getElementById("report").innerHTML = `<span style="color:red;"> Error: ${err.message}</span>`;
    }
}

// ============================
// 📋 Summary Report Loader
// ============================

async function loadSummary() {
    try {
      const res = await fetch("/summary");
      const summary = await res.json();
  
      let html = `<table>
        <tr>
          <th>Word</th>
          <th>HNSW Time</th><th>HNSW Steps</th><th>HNSW Cosine Similarity</th>
          <th>ACORN Time</th><th>ACORN Steps</th><th>ACORN Cosine Similarity</th>
        </tr>`;
  
      let total = { h_time: 0, h_steps: 0, h_sim: 0, a_time: 0, a_steps: 0, a_sim: 0 };
  
      summary.forEach(row => {
        html += `<tr>
          <td>${row.word}</td>
          <td>${row.hnsw.time_ms} ms</td><td>${row.hnsw.steps}</td><td>${row.hnsw.sim}</td>
          <td>${row.acorn.time_ms} ms</td><td>${row.acorn.steps}</td><td>${row.acorn.sim}</td>
        </tr>`;
  
        total.h_time += row.hnsw.time_ms;
        total.h_steps += row.hnsw.steps;
        total.h_sim += row.hnsw.sim;
        total.a_time += row.acorn.time_ms;
        total.a_steps += row.acorn.steps;
        total.a_sim += row.acorn.sim;
      });
  
      const count = summary.length;
      const avgRow = `
        <tr style="font-weight: bold;">
          <td>Average</td>
          <td>${(total.h_time / count).toFixed(2)} ms</td>
          <td>${(total.h_steps / count).toFixed(2)}</td>
          <td>${(total.h_sim / count).toFixed(4)}</td>
          <td>${(total.a_time / count).toFixed(2)} ms</td>
          <td>${(total.a_steps / count).toFixed(2)}</td>
          <td>${(total.a_sim / count).toFixed(4)}</td>
        </tr>`;
  
      html += avgRow;
      document.getElementById("summary-table").innerHTML = html;
  
      // 👆 Show averages above table
      const avgText = `
        <strong> Averages:</strong><br>
        HNSW: ${ (total.h_time / count).toFixed(2) } ms, ${ (total.h_steps / count).toFixed(2) } steps, Sim ${ (total.h_sim / count).toFixed(4) }<br>
        ACORN-1: ${ (total.a_time / count).toFixed(2) } ms, ${ (total.a_steps / count).toFixed(2) } steps, Sim ${ (total.a_sim / count).toFixed(4) }
      `;
      document.getElementById("summary-averages").innerHTML = avgText;
  
    } catch (err) {
      document.getElementById("summary-table").innerHTML = `<p style="color:red;">Error loading summary: ${err.message}</p>`;
    }
  }
  

  
  document.getElementById("refresh-summary-btn").addEventListener("click", () => {
    loadSummary();
  });
  
  
