import * as d3 from 'd3';

function linkArc(d) {
    const r = Math.hypot(d.target.x - d.source.x, d.target.y - d.source.y);
    const offsetX = ((d.target.x - d.source.x) * d.target.size) / r;
    const offsetY = ((d.target.y - d.source.y) * d.target.size) / r;
    return `
    M${d.source.x},${d.source.y}
    A${r},${r} 0 0,1 ${d.target.x - offsetX},${d.target.y - offsetY}
  `;
}

function drag(simulation) {

    function dragstarted(event) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
    }

    function dragged(event) {
        event.subject.fx = event.x;
        event.subject.fy = event.y;
    }

    function dragended(event) {
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
    }

    return d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended);
}

const draw = (props) => {
    const node_data = props.node_data;
    const edge_data = props.edge_data;
    const links = edge_data.map(d => Object.create(d));
    const nodes = node_data.map(d => Object.create(d));
    const simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id(d => d.id))
        .force("charge", d3.forceManyBody().strength(-400))
        .force("x", d3.forceX())
        .force("y", d3.forceY());

    d3.select('#vis-diagramchart').selectAll("*").remove();
    const margin = { top: 10, right: 100, bottom: 20, left: 10 };
    const width = props.width - margin.left - margin.right;
    const height = props.height - margin.top - margin.bottom;

    let color = d3.scaleLinear().domain([0,0.8,1])
        .range(["#f50057", "#673ab7", "#2196f3"])

    let svg = d3.select('.vis-diagramchart')
        .append('svg')
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [-width / 2, -height / 2, props.width, props.height])
        .append('g');

    // Create a tooltip div that is hidden by default:
    let tooltip = d3.select(".vis-diagramchart")
        .append("div")
        .style("opacity", 0)
        .attr("class", "tooltip")
        .style("background-color", "#f0f2f5")
        .style("border", "solid")
        .style("border-width", "1px")
        .style("border-radius", "5px")
        .style("padding", "5px")
        .style("color", "black")
    
    let showTooltip = function(d) {
        tooltip
        .transition()
        .duration(200)
        tooltip
        .style("opacity", 1)
        .html("Node: " + d.id + ', Sincere: ' + d.positive + ', Insincere: ' + d.negative)
        .style("position", "absolute")
        .style("z-index", 10)
    }
    let moveTooltip = function(d) {
        tooltip
        .style("top", (d3.event.pageY-10)+"px")
        .style("left",(d3.event.pageX+10)+"px")
    }
    let hideTooltip = function(d) {
        tooltip
        .transition()
        .duration(200)
        .style("opacity", 0)
    }

    svg.append("defs")
        .append('marker')
        .attr("id", "end-arrow")
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 12)
        .attr("refY", -0.5)
        .attr("markerWidth", 3)
        .attr("markerHeight", 3)
        .attr("orient", "auto")
        .style("fill", "grey")
        .append("path")
        .attr("d", "M0,-5L10,0L0,5");

    let link = svg.selectAll("path")
        .data(links)
        .enter()
        .append('path')
        .attr("fill", "none")
        .attr("stroke-width", d => d.width) // edge width
        .attr("stroke", "gray")
        .attr("marker-end", "url(#end-arrow)");

    const node = svg.append("g")
        .attr("fill", "currentColor")
        .attr("stroke-linecap", "round")
        .attr("stroke-linejoin", "round")
        .selectAll("g")
        .data(nodes)
        .enter()
        .append('g')
        .on("mouseover", showTooltip )
        .on("mousemove", moveTooltip )
        .on("mouseleave", hideTooltip )
        .style('visibility', function (d) {
            return d.visible !== true ? "hidden" : "visible";
        })
        // .call(drag(simulation));

    function clicked(event, d) {
        if (nodes[d].id.toString() !== '0') {
            props.changeSelectState(nodes[d].id.toString());
        }
    }

    node.append("circle")
        .attr("fill", d => color((d.positive / (d.positive + d.negative))))
        .attr("stroke", "white")
        .attr("r", d => d.size)
        .on("click", clicked)

    node.append("text")
        .attr("x", 10)
        .attr("y", "0.31em")
        .text(d => d.id)
        .style("font-size", "20px")
        .style("fill", "#004669")
        .style("font-weight", "bold")

    simulation.on("tick", () => {
        link.attr("d", linkArc);
        node.attr("transform", d => `translate(${d.x},${d.y})`);
    });

}

export default draw;