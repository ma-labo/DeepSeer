import React, { Component } from 'react';
import './view4.css';
import DiagramChart from '../../charts/DiagramChart';

function create_edge_data(trace) {
    var edge_data = [{'source': 0, 'target': trace[0], 'width': 1}];
    for (let i = 1; i < trace.length; i++) {
        var obj = {
            'source': trace[i - 1],
            'target': trace[i],
            'width': 1
        }
        if (trace[i - 1] !== trace[i]) {
            edge_data.push(obj)
        }
    }
    return edge_data
}

function filter_node_data(trace, node_data) {
    let filtered_node_data = []
    let visible_nodes = trace === '' ? null : new Set(trace)
    for (let i = 0; i < node_data.length; i++) {
        if (visible_nodes === null || visible_nodes.has(node_data[i].id)) {
            filtered_node_data.push({
                'id': node_data[i].id,
                'size': node_data[i].size,
                'world': node_data[i].world,
                'sports': node_data[i].sports,
                'business': node_data[i].business,
                'scitech': node_data[i].scitech,
                'visible': true
            })
        }
        else {
            filtered_node_data.push({
                'id': node_data[i].id,
                'size': node_data[i].size,
                'world': node_data[i].world,
                'sports': node_data[i].sports,
                'business': node_data[i].business,
                'scitech': node_data[i].scitech,
                'visible': false
            })
        }
    }
    return filtered_node_data;
}

export default class DiagramView extends Component {
    render() {
        const trace = this.props.trace
        const node_data = filter_node_data(trace, require('../../data/agnews/node_data.json'))
        const edge_data = trace === '' ? require('../../data/agnews/edge_data.json') : create_edge_data(trace)
        const width = 1000
        const height = 600;
        return (
            <div id='view4' className='pane' >
                <div>
                    <DiagramChart node_data={node_data} edge_data={edge_data} width={width} height={height}
                                  changeSelectState={this.props.changeSelectState}/>
                </div>
            </div>
        )
    }
}