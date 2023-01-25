import React, { Component } from 'react';
import draw from './vis';

export default class DiagramChart extends Component {

    componentDidMount() {
        draw(this.props);
    }

    shouldComponentUpdate(nextProps, nextState, nextContext) {
        return (nextProps.edge_data !== this.props.edge_data)
    }

    componentDidUpdate(preProps) {
        draw(this.props);
    }

    render() {
        return (
            <div id="vis-diagramchart" className='vis-diagramchart'/>
        )
    }
}

