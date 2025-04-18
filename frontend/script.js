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
            name: `Layer ${layer}`,
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
                name: `HNSW Path (Layer ${layer})`,
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

        Plotly.newPlot("traversal-plot", [
            ...layerTracesPlot2,
            queryTrace,
            entryTrace,
            ...hnswLayerTraces,
            acornTrace
        ], {
            title: {
                text: `Search Traversal for '${data.query}'`,
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
            <strong>🔍 Query:</strong> ${data.query}<br><br>
            <strong>HNSW Result:</strong> ${data.hnsw.result}<br>
            ⏱ Time: ${data.hnsw.time_ms}ms | 🔁 Steps: ${data.hnsw.num_visited} <br>
            📐 Cosine Sim: ${data.hnsw.similarity.toFixed(4)}<br><br>
            <strong>ACORN-1 Result:</strong> ${data.acorn.result}<br>
            ⏱ Time: ${data.acorn.time_ms}ms | 🔁 Steps: ${data.acorn.num_visited}<br>
            📐 Cosine Sim: ${data.acorn.similarity.toFixed(4)}
        `;

    } catch (err) {
        console.error("Error occurred:", err);
        document.getElementById("report").innerHTML = `<span style="color:red;">❌ Error: ${err.message}</span>`;
    }
};
