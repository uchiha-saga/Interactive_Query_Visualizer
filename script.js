document.addEventListener("DOMContentLoaded", function() {
    // ========== SELECT ELEMENTS ==========
    const svg = d3.select("#plan-svg"),
          svgWidth = +svg.attr("width"),
          svgHeight = +svg.attr("height");
    
    const uploadArea = document.getElementById("upload-area");
    const downloadBtn = document.getElementById("download-json");
    const totalTimeEl = document.getElementById("total-time");
    const rowsReturnedEl = document.getElementById("rows-returned");
    const rowsScannedEl = document.getElementById("rows-scanned");
    const prominentOpsEl = document.getElementById("prominent-ops");
    const executionTimelineEl = document.getElementById("execution-timeline");
    const nodeDetailsEl = document.getElementById("node-details"); // New panel for node details
    
    // ========== DEFINE MARGINS & LAYOUT ==========
    const margin = { top: 20, right: 20, bottom: 20, left: 100 };
    const width = svgWidth - margin.left - margin.right;
    const height = svgHeight - margin.top - margin.bottom;
    
    const treeLayout = d3.tree().size([height, width]);
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10);
    
    // ========== TOOLTIP SETUP ==========
    const tooltip = d3.select("body")
                      .append("div")
                      .attr("class", "tooltip")
                      .style("opacity", 0);
    
    // ========== DATA STORAGE ==========
    let currentPlanJson = null;
    let globalRoot = null; // Will hold our D3 hierarchy
    
    // ========== HELPER FUNCTIONS ==========
    
    // Convert the raw JSON plan into a hierarchical structure
    function convertPlanToTree(node) {
      const label = `${node["Node Type"]}${node["Relation Name"] ? " (" + node["Relation Name"] + ")" : ""}`;
      const details = `Rows: ${node["Actual Rows"] || "?"}\nTime: ${
        node["Actual Total Time"] ? node["Actual Total Time"].toFixed(2) + "ms" : "?"
      }`;
      const children = (node.Plans || []).map(convertPlanToTree);
      return {
        name: label,
        details: details,
        type: node["Node Type"],
        time: node["Actual Total Time"] || 0,
        rows: node["Actual Rows"] || 0,
        children: children.length > 0 ? children : null
      };
    }
    
    // Update the node details panel with information about the clicked node
    function updateNodeDetails(d) {
      nodeDetailsEl.innerHTML = `
        <p><strong>Node Type:</strong> ${d.data.type}</p>
        <p><strong>Label:</strong> ${d.data.name}</p>
        <p><strong>Details:</strong><br/>${d.data.details.replace(/\n/g, "<br/>")}</p>
      `;
    }
    
    // Collapse a node and its children
    function collapse(d) {
      if (d.children) {
        d._children = d.children;
        d._children.forEach(collapse);
        d.children = null;
      }
    }
    
    // Update (or re-render) the tree with collapsible behavior
    function update(source) {
      // Compute the new tree layout.
      const root = globalRoot;
      treeLayout(root);
      
      // Normalize for fixed depth.
      root.descendants().forEach(d => { d.y = d.depth * 180; });
      
      // Clear the previous contents.
      svg.selectAll("*").remove();
      const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
      
      // Links
      const links = root.links();
      g.selectAll(".link")
        .data(links)
        .enter()
        .append("path")
        .attr("class", "link")
        .attr("d", d3.linkHorizontal()
                     .x(d => d.y)
                     .y(d => d.x)
        );
      
      // Nodes
      const nodes = root.descendants();
      const node = g.selectAll(".node")
                    .data(nodes)
                    .enter()
                    .append("g")
                    .attr("class", "node")
                    .attr("transform", d => `translate(${d.y},${d.x})`)
                    .on("click", function(event, d) {
                      // Toggle children
                      if (d.children) {
                        d._children = d.children;
                        d.children = null;
                      } else {
                        d.children = d._children;
                        d._children = null;
                      }
                      update(d);
                      updateNodeDetails(d);
                    })
                    .on("mouseover", (event, d) => {
                      tooltip.transition().duration(200).style("opacity", 0.9);
                      tooltip.html(`<strong>${d.data.type}</strong><br/>${d.data.details.replace(/\n/g, "<br/>")}`)
                             .style("left", (event.pageX + 10) + "px")
                             .style("top", (event.pageY - 28) + "px");
                    })
                    .on("mouseout", () => {
                      tooltip.transition().duration(500).style("opacity", 0);
                    });
      
      node.append("circle")
          .attr("r", 8)
          .attr("fill", d => colorScale(d.data.type));
      
      node.append("text")
          .attr("dy", 3)
          .attr("x", d => (d.children || d._children) ? -12 : 12)
          .style("text-anchor", d => (d.children || d._children) ? "end" : "start")
          .text(d => d.data.name);
    }
    
    // Update query summary, prominent operators, and timeline (same as before)
    function updateSummary(planRoot) {
      const totalTime = planRoot["Actual Total Time"] || 0;
      const rowsReturned = planRoot["Actual Rows"] || 0;
      let totalScanned = 0;
      function sumRows(node) {
        totalScanned += (node["Actual Rows"] || 0);
        if (node.Plans) { node.Plans.forEach(child => sumRows(child)); }
      }
      sumRows(planRoot);
      
      totalTimeEl.textContent = totalTime.toFixed(2);
      rowsReturnedEl.textContent = rowsReturned;
      rowsScannedEl.textContent = totalScanned;
      
      let nodeList = [];
      function collectNodes(node) {
        nodeList.push({
          type: node["Node Type"],
          time: node["Actual Total Time"] || 0
        });
        if (node.Plans) { node.Plans.forEach(child => collectNodes(child)); }
      }
      collectNodes(planRoot);
      nodeList.sort((a, b) => b.time - a.time);
      const topOps = nodeList.slice(0, 5);
      
      prominentOpsEl.innerHTML = topOps.slice(0, 3).map(op =>
        `<p>${op.type}: ${op.time.toFixed(2)} ms</p>`
      ).join("");
      
      buildTimeline(topOps, executionTimelineEl);
    }
    
    // Build a horizontal bar chart timeline (same as before)
    function buildTimeline(nodeList, container) {
      container.innerHTML = "";
      if (nodeList.length === 0) {
        container.innerHTML = "<p>No operators to display.</p>";
        return;
      }
      const timelineWidth = 300;
      const timelineHeight = 150;
      const marginTimeline = { top: 20, right: 20, bottom: 30, left: 100 };
      const widthTimeline = timelineWidth - marginTimeline.left - marginTimeline.right;
      const heightTimeline = timelineHeight - marginTimeline.top - marginTimeline.bottom;
      
      const timelineSvg = d3.select(container)
        .append("svg")
        .attr("width", timelineWidth)
        .attr("height", timelineHeight);
      
      const gTimeline = timelineSvg.append("g")
        .attr("transform", `translate(${marginTimeline.left},${marginTimeline.top})`);
      
      const maxTime = d3.max(nodeList, d => d.time);
      const x = d3.scaleLinear().domain([0, maxTime]).range([0, widthTimeline]);
      const y = d3.scaleBand().domain(nodeList.map(d => d.type)).range([0, heightTimeline]).padding(0.1);
      
      gTimeline.append("g")
        .attr("transform", `translate(0,${heightTimeline})`)
        .call(d3.axisBottom(x));
      
      gTimeline.append("g")
        .call(d3.axisLeft(y));
      
      gTimeline.selectAll(".bar")
        .data(nodeList)
        .enter()
        .append("rect")
        .attr("class", "bar")
        .attr("y", d => y(d.type))
        .attr("x", 0)
        .attr("height", y.bandwidth())
        .attr("width", d => x(d.time))
        .attr("fill", "#3498db");
      
      gTimeline.selectAll(".label")
        .data(nodeList)
        .enter()
        .append("text")
        .attr("class", "label")
        .attr("x", d => x(d.time) + 4)
        .attr("y", d => y(d.type) + y.bandwidth() / 2 + 4)
        .text(d => d.time.toFixed(2) + " ms")
        .attr("fill", "#333");
    }
    
    // ========== MAIN LOGIC ==========
    
    function handleFile(file) {
      const reader = new FileReader();
      reader.onload = function(e) {
        let json;
        try {
          json = JSON.parse(e.target.result);
        } catch (error) {
          alert("Error parsing JSON file.");
          return;
        }
        // Expect a top-level "Plan" property or an array with a "Plan" key
        const plan = json.Plan || (Array.isArray(json) && json[0] && json[0].Plan);
        if (!plan) {
          alert("Invalid query_plan.json format.");
          return;
        }
        currentPlanJson = json;
        const treeData = convertPlanToTree(plan);
        globalRoot = d3.hierarchy(treeData);
        // Initially collapse all children
        if (globalRoot.children) {
          globalRoot.children.forEach(collapse);
        }
        update(globalRoot);
        updateSummary(plan);
      };
      reader.readAsText(file);
    }
    
    // ========== EVENT HANDLERS ==========
    
    uploadArea.addEventListener("dragover", e => {
      e.preventDefault();
      uploadArea.style.background = "#eee";
    });
    uploadArea.addEventListener("dragleave", () => {
      uploadArea.style.background = "";
    });
    uploadArea.addEventListener("drop", e => {
      e.preventDefault();
      uploadArea.style.background = "";
      handleFile(e.dataTransfer.files[0]);
    });
    uploadArea.addEventListener("click", () => {
      const input = document.createElement("input");
      input.type = "file";
      input.accept = ".json";
      input.onchange = e => handleFile(e.target.files[0]);
      input.click();
    });
    
    downloadBtn.addEventListener("click", () => {
      if (!currentPlanJson) {
        alert("No plan loaded yet!");
        return;
      }
      const blob = new Blob([JSON.stringify(currentPlanJson, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "query_plan.json";
      a.click();
      URL.revokeObjectURL(url);
    });
  });
  