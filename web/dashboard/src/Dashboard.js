import React, { Component } from 'react';
// import data from './data';
import { Layout } from 'antd';
import {
    escapeRegExp,
} from '@material-ui/data-grid';
import Button from '@material-ui/core/Button';
import Box from '@material-ui/core/Box';
import QuickFilteringGrid from "./views/instance_view/DataGrid";
import DiagramView from "./views/diagram_view";
import NestedList from "./views/pattern_summary_view/NestedList";
import './dashboard.css';
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import Typography from '@material-ui/core/Typography';
import IconButton from '@material-ui/core/IconButton';
import MenuIcon from '@material-ui/icons/Menu';
import CopyrightIcon from '@material-ui/icons/Copyright';
import Popover from '@material-ui/core/Popover';
import InputBase from '@material-ui/core/InputBase';
import SearchIcon from '@material-ui/icons/Search';
import Paper from '@material-ui/core/Paper';
import CreateIcon from '@material-ui/icons/Create';
import axios from 'axios'
import Grid from '@material-ui/core/Grid';
import BackspaceIcon from '@material-ui/icons/Backspace'
import ClearIcon from '@material-ui/icons/Clear';
import Chip from '@material-ui/core/Chip';
import Avatar from '@material-ui/core/Avatar';
import SentimentSatisfiedAltIcon from "@material-ui/icons/SentimentSatisfiedAlt";
import SentimentVeryDissatisfiedIcon from "@material-ui/icons/SentimentVeryDissatisfied";
import Card from '@material-ui/core/Card';
import CardActions from '@material-ui/core/CardActions';
import CardContent from '@material-ui/core/CardContent';

const { Sider, Content, Footer } = Layout;
const buggy_pattern = require('./data/quora/buggy_pattern.json')
const sentimental_pattern = require('./data/quora/sentimental_pattern.json')
const state_words = require('./data/quora/state_words.json')

function removePound(str) {
    let new_str = ''
    let word_array = str.split(' ')
    if ((word_array[0][0] === '#') && word_array[0][1] === '#') {
        new_str = word_array[0].slice(2, word_array[0].length)
    }
    else {
        new_str = word_array[0]
    }
    for (let i = 1; i < word_array.length; i++) {
        new_str += ' '
        if ((word_array[i][0] === '#') && word_array[i][1] === '#') {
            new_str = new_str.slice(0, new_str.length - 1)
            new_str += (word_array[i].slice(2, word_array[i].length))
        }
        else {
            new_str += (word_array[i])
        }
    }
    return new_str
}

const ColourfulTrace = function ColourfulTrace(props) {
    const { value, seq_label, label } = props;
    const symbol = value.includes('➞') ? '➞' : ' '
    const items = value.split(symbol);
    return (
        <div>
            <Box>
                {
                    symbol === '➞' ?
                        <Typography variant="subtitle1" color="textPrimary" display='inline'>
                            Trace:
                        </Typography> :
                        <Typography variant="subtitle1" color="textPrimary" display='inline'>
                            Input text:
                        </Typography>
                }
                <Typography variant="subtitle1" display='inline' color={seq_label[0] === 0 ? 'primary' : 'secondary'} style={{ wordWrap: "break-word"}}>
                    {' ' + items.shift()}
                </Typography>
                {
                    items.map((item, index) => {
                        return (
                            <Box display='inline'>
                                <Typography variant="subtitle1" display='inline' style={{wordWrap: "break-word", color: "grey"}}>
                                    {symbol}
                                </Typography>
                                <Typography variant="subtitle1" display='inline' color={seq_label.slice(1, seq_label.length)[index] === 0 ? 'primary' : 'secondary'} style={{wordWrap: "break-word"}}>
                                    {item.slice(0,2) === '##' ? item.slice(2, item.length) : item}
                                </Typography>
                            </Box>
                        )
                    })
                }
                {'  '}
                {
                    label === true ?
                        <Button
                            variant="outlined"
                            style={{borderColor: (seq_label[seq_label.length - 1] === 0 ? '#2196f3' : '#b26500'),
                                color: (seq_label[seq_label.length - 1] === 0 ? '#2196f3' : '#b26500')}}
                            size="small"
                            endIcon={seq_label[seq_label.length - 1] === 0 ? <SentimentSatisfiedAltIcon /> : <SentimentVeryDissatisfiedIcon /> }
                            disabled={true}
                        >
                            {seq_label[seq_label.length - 1] === 0 ? 'Sincere' : 'Insincere'}
                        </Button> : null
                }
            </Box>
        </div>
    )
};

function checkIfContainBug(trace_to_plot) {
    let possible_buggy_pattern = new Set()
    for (let i = 0; i < buggy_pattern.length; i ++) {
        let pattern = buggy_pattern[i].key;
        for (let j = 0; j < trace_to_plot.length - pattern.length; j ++) {
            if (trace_to_plot[j] === pattern[0]) {
                let k = 1;
                while ((trace_to_plot[j + k] === pattern[k]) && (k < pattern.length)) {
                    k += 1;
                }
                if (k === pattern.length) {
                    possible_buggy_pattern.add(pattern.join(' ') + '_' + i.toString())
                }
            }
        }
    }
    return Array.from(possible_buggy_pattern);
}

function checkIfContainSentiment(trace_to_plot, seq_labels, text) {
    let possible_buggy_pattern = new Set()
    for (let i = 0; i < sentimental_pattern.length; i ++) {
        let pattern = sentimental_pattern[i].key;
        for (let j = 0; j < trace_to_plot.length - pattern.length + 1; j ++) {
            if (trace_to_plot[j] === pattern[0]) {
                let k = 1;
                while ((trace_to_plot[j + k] === pattern[k]) && (k < pattern.length)) {
                    k += 1;
                }
                let sentimental = seq_labels.slice(j, j + k).reduce((a, b) => a + b)
                if ((k === pattern.length) && (sentimental > 0) && (sentimental < k)) {
                    for (let w = 0; w < sentimental_pattern[i].mined_results.length; w ++){
                        if (sentimental_pattern[i].mined_results[w].text === text.slice(j, j + k).join(' ')) {
                            possible_buggy_pattern.add(pattern.join(' ') + '_' + i.toString() + '_' + text.slice(j, j + k).join(' '))
                        }
                    }
                }
            }
        }
    }
    return Array.from(possible_buggy_pattern);
}

function gatherStateWords(selectedState) {
    if (selectedState !== 'all') {
        for (let i = 0; i < state_words.length; i++) {
            if (state_words[i].state.toString() === selectedState) {
                let associated_words = state_words[i].words[0];
                return {
                    'in_state': associated_words[0],
                    'words': associated_words[1]
                }
            }
        }
    }
    else {
        return null;
    }
}

function getRegexPattern(selectedStates) {
    let regexPattern = ''
    let splitStates = selectedStates.split(' ')
    let regexPatternArray = ['([\\D]' + splitStates[0] + '[\\D]|[\\D]' + splitStates[0] + '$|^' + splitStates[0] + '[\\D])']
    for (let i = 1; i < splitStates.length - 1; i++) {
        regexPatternArray.push('(' + splitStates[i] + '[\\D])')
    }
    if (splitStates.length > 1) {
        regexPatternArray.push('(' + splitStates[splitStates.length-1] + '[\\D]|' + splitStates[splitStates.length-1] + '$)')
    }
    regexPattern = regexPatternArray.join('+.*')
    regexPattern += '+'
    return regexPattern
}

export default class Dashboard extends Component {
    constructor(props) {
        super(props);
        this.state = {
            selectedStates: 'all', // '3 25 34'
            selectedEdges: 'all',
            selectedTrace: '',
            selectedPatterns: 'all',
            popoverOpen: null,
            textInput: '',
            deepStellarResult: null,
            isFetching: false,
            dataset: 'train',
            patternSelectedIndex: -1,
            searchText: ''
        }
    }

    changeSelectState = value => {
        this.setState({
            selectedStates: this.state.selectedStates === value ? 'all' : value
        })
    }

    pushState = value => {
        if (this.state.selectedStates === 'all') {
            this.setState({
                selectedStates: value
            })
        }
        else {
            const tmp = this.state.selectedStates + ' ' + value
            this.setState({
                selectedStates: tmp
            })
        }
    }

    changeSelectEdge = value => {
        this.setState({
            selectedEdges: value
        })
    }

    changeSelectPattern = value => {
        this.setState({
            selectedPatterns: value
        })
    }

    changeSelectTrace = value => {
        this.setState({
            selectedTrace: value
        })
    }

    handleClick = (event) => {
        this.setState({
            popoverOpen: event.currentTarget
        })
    }

    handleClose = (event) => {
        this.setState({
            popoverOpen: null
        })
    }

    handleClear = () => {
        this.changeSelectState('all')
        this.changePatternSelectedIndex(-1)
        this.changeSelectPattern('all')
        this.changeSearchText('')
        this.setState({
            deepStellarResult: null
        })
    }

    handleTextInput = (event) => {
        this.setState({
            textInput: event.target.value
        })
    }

    handleSendingText = () => {
        console.log(this.state.textInput)
        if (!this.state.textInput) {
            return;
        }
        this.setState({...this.state, isFetching: true})
        axios.get('http://127.0.0.1:5000/api/v1/predict?input=' + this.state.textInput)
            .then((response) => {
                console.log("check result ", response.data);
                this.setState({...this.state, deepStellarResult: response.data, isFetching: false, popoverOpen: null}) //, selectedStates: response.data['result']['states'].join(' ')})
            })
            .catch(e => {
                console.log(e);
                this.setState({...this.state, isFetching: false});
            });

    }

    handleDatasetChange = () => {
        this.state.dataset === 'train' ? this.setState({dataset: 'test'}) : this.setState({dataset: 'train'});
    };

    changePatternSelectedIndex = (value) => {
        this.setState({
            patternSelectedIndex: value
        })
    }

    changeSearchText = (value) => {
        this.setState({
            searchText: value
        })
    }

    render() {
        const data_rows = this.state.dataset === 'train' ? require('./data/quora/training_data.json') : require('./data/quora/test_data.json');
        const {selectedStates, selectedEdges, selectedPatterns, selectedTrace, textInput, deepStellarResult, searchText} = this.state;
        const searchRegexPatterns = new RegExp(selectedPatterns.split('_')[0], 'i');
        const regexPattern = selectedStates === 'all' ? '' : getRegexPattern(selectedStates)
        const searchRegexStates = new RegExp(regexPattern, 'i');
        const filteredRows = searchText !== '' ?
            data_rows.filter((row) => {return RegExp(searchText, 'i').test(row['text'].join(' '))}) :
            selectedPatterns.split('_')[0] !== 'all' ?
                data_rows.filter((row) => {return searchRegexPatterns.test(row['trace'].join(' ').toString());}) :
                (selectedStates !== 'all' ?
                    data_rows.filter((row) => {return searchRegexStates.test(row['trace'].join(' ').toString());}) :
                    data_rows)
        const traceToPlot = deepStellarResult !== null ? deepStellarResult['result']['input_trace'] : (selectedTrace === '' ? '' : (selectedTrace.length > 0 ? data_rows[selectedTrace].trace: ''))
        const seqlabelToPlot = deepStellarResult !== null ? deepStellarResult['result']['seq_label'] : (selectedTrace === '' ? '' : (selectedTrace.length > 0 ? data_rows[selectedTrace].seq_label: ''))
        const textToPlot = deepStellarResult !== null ? deepStellarResult['result']['text'] : (selectedTrace === '' ? '' : (selectedTrace.length > 0 ? data_rows[selectedTrace].text: ''))
        const possibleBuggyPattern = traceToPlot === '' ? [] : checkIfContainBug(traceToPlot)
        const possibleSentimentalPattern = traceToPlot === '' ? [] : checkIfContainSentiment(traceToPlot, seqlabelToPlot, textToPlot)
        // console.log(this.state.searchText)
        // console.log(possibleBuggyPattern)
        // const filteredGraph = selectedTrace !== '' ? xxx : xxx
        const associated_words = gatherStateWords(selectedStates)

        const open = Boolean(this.state.popoverOpen);
        const id = open ? 'simple-popover' : undefined;
        return (
            <div>
                <AppBar position="static">
                    <Toolbar>
                        <IconButton edge="start" color="inherit" aria-label="menu">
                            <MenuIcon />
                        </IconButton>
                        <Typography variant="h4" color="inherit">
                            DeepSeer
                        </Typography>
                    </Toolbar>
                </AppBar>
                <Layout style={{ height: 1200, backgroundColor: '#f9f9f9' }}>
                    <Layout style={{ height: 800, backgroundColor: '#f9f9f9' }}>
                        <Content>
                            <div>
                                <Box p={1} ml={2}>
                                    <Typography variant="h5" color="primary">
                                        State Diagram
                                    </Typography>
                                    {/*{(deepStellarResult === null) && (traceToPlot === '') ?*/}
                                    {/*    <Box component="div" display="inline" mr={1}>*/}
                                    {/*        <Typography variant="subtitle1" color="textPrimary" display="inline">*/}
                                    {/*            Selected pattern:*/}
                                    {/*        </Typography>*/}
                                    {/*        {' '}*/}
                                    {/*        {*/}
                                    {/*            selectedPatterns !== 'all' ?*/}
                                    {/*                <Chip*/}
                                    {/*                    variant="outlined"*/}
                                    {/*                    size='small'*/}
                                    {/*                    avatar={<Avatar>{selectedPatterns.split('_')[1]}</Avatar>}*/}
                                    {/*                    label={selectedPatterns.split('_')[0]}*/}
                                    {/*                    color="secondary"*/}
                                    {/*                /> : <Typography variant="subtitle1" color="textPrimary" display="inline">*/}
                                    {/*                    all*/}
                                    {/*                </Typography>*/}
                                    {/*        }*/}
                                    {/*    </Box> :*/}
                                    {/*    null}*/}
                                </Box>
                                {/*<Box p={1} ml={2}>*/}
                                {/*    {(possibleBuggyPattern.length !== 0 && deepStellarResult !== null) ? possibleBuggyPattern.map((pattern) => {*/}
                                {/*        return (*/}
                                {/*            <Box component="div" display="inline" m={0.5}>*/}
                                {/*                <Chip*/}
                                {/*                    variant="outlined"*/}
                                {/*                    size='small'*/}
                                {/*                    avatar={<Avatar>{pattern.split('_')[1]}</Avatar>}*/}
                                {/*                    label={pattern.split('_')[0]}*/}
                                {/*                    color="secondary"*/}
                                {/*                    // clickable*/}
                                {/*                    // onClick={(event) => this.changePatternSelectedIndex(pattern.split('_')[0])}*/}
                                {/*                />*/}
                                {/*            </Box>*/}
                                {/*        )*/}
                                {/*    }) : null}*/}
                                {/*</Box>*/}
                                <Box p={1} ml={2}>
                                    {possibleSentimentalPattern.length !== 0 ? possibleSentimentalPattern.map((pattern) => {
                                        return (
                                            <Box component="div" display="inline" m={0.5}>
                                                <Chip
                                                    variant="outlined"
                                                    size='small'
                                                    avatar={<Avatar>{pattern.split('_')[1]}</Avatar>}
                                                    label={pattern.split('_')[0]}
                                                    color="primary"
                                                    // clickable
                                                    // onClick={(event) => this.changePatternSelectedIndex(pattern.split('_')[0])}
                                                />
                                            </Box>
                                        )
                                    }) : null}
                                </Box>
                            </div>
                            <div>
                                <Grid container spacing={0}>
                                    <Grid item xs={3}>
                                        <Box p={1} ml={2}>
                                            <Box pt={1} pb={1}>
                                                {/*<Button size="small" color="primary" variant="outlined" endIcon={<BackspaceIcon />}*/}
                                                {/*        onClick={() => {*/}
                                                {/*            if (selectedStates !== 'all') {*/}
                                                {/*                let i = selectedStates.length - 1*/}
                                                {/*                while((selectedStates[i] !== ' ') && (i > 0)) {*/}
                                                {/*                    i -= 1;*/}
                                                {/*                }*/}
                                                {/*                this.changeSelectState(i > 0 ? selectedStates.substr(0, i) : 'all')*/}
                                                {/*            }*/}
                                                {/*        }}>Back</Button>*/}
                                                {/*{'  '}*/}
                                                <Button size="small" color="secondary" variant="outlined" endIcon={<ClearIcon/>}
                                                        onClick={this.handleClear}>Clear</Button>
                                            </Box>
                                        </Box>
                                    </Grid>
                                    <Grid item x={7}>
                                        <Box p={1} ml={2}>
                                            <Box pt={1} pb={1}>
                                                <Chip label="Sincere" color="primary" style={{maxWidth: '80px', minWidth: '80px'}}/>
                                                {' '}
                                                <Chip label="Insincere" color="secondary" style={{maxWidth: '80px', minWidth: '80px'}}/>
                                            </Box>
                                        </Box>
                                    </Grid>
                                </Grid>
                            </div>
                            <Grid container spacing={0}>
                                <Grid item xs={3}>
                                    <Box p={2}>
                                        <Paper component="form" elevation={2}>
                                            <Box ml={2} mr={1} display="flex">
                                                <InputBase
                                                    placeholder="Type to inference a new sentence..."
                                                    inputProps={{ 'aria-label': 'search new sentence' }}
                                                    fullWidth={true}
                                                    multiline={true}
                                                    minRows={5}
                                                    maxRows={10}
                                                    onChange={this.handleTextInput}
                                                />
                                                <IconButton onClick={this.handleSendingText} disabled={this.state.textInput === ''}>
                                                    <SearchIcon />
                                                </IconButton>
                                            </Box>
                                        </Paper>
                                    </Box>
                                    {(deepStellarResult === null) && (traceToPlot === '') ?
                                        null :
                                        <Box p={3}>
                                            <ColourfulTrace value={textToPlot.join(' ')} seq_label={seqlabelToPlot} label={true}/>
                                            <ColourfulTrace value={traceToPlot.join('➞')} seq_label={seqlabelToPlot}/>
                                        </Box>}
                                </Grid>
                                <Grid item xs={7}>
                                    <DiagramView trace={traceToPlot} changeSelectState={this.changeSelectState}/>
                                </Grid>
                                <Grid item xs={2}>
                                    <Box p={2}>
                                        {selectedStates !== 'all' ?
                                            <Card variant='outlined'>
                                                <CardContent>
                                                    <Typography color="textSecondary" gutterBottom>
                                                        State: {selectedStates}
                                                    </Typography>
                                                    <Typography variant="h7" component="h7" style={{ fontWeight: 600 }}>
                                                        associated words and phrases
                                                    </Typography>
                                                    <Grid container spacing={0}>
                                                        <Grid item xs={6}>
                                                            <Typography variant="h5" component="h2" color='primary'>
                                                                {removePound(associated_words.words[0][0])}
                                                            </Typography>
                                                        </Grid>
                                                        <Grid item xs={6}>
                                                            <Chip label={'count: ' + associated_words.words[0][1]} variant="outlined" size="small" />
                                                        </Grid>
                                                    </Grid>
                                                    {/*<Typography color="textPrimary">*/}
                                                    {/*    {associated_words.in_state.toString() + '➞' + selectedStates.toString()}*/}
                                                    {/*</Typography>*/}
                                                    {associated_words.words.slice(1, Math.min(associated_words.words.length, 3)).map((value) => {
                                                        return (
                                                            <Grid container spacing={0}>
                                                                <Grid item xs={6}>
                                                                    <Typography variant="h6" component="p">
                                                                        {removePound(value[0])}
                                                                    </Typography>
                                                                </Grid>
                                                                <Grid item xs={6}>
                                                                    <Chip label={'count: ' + value[1]} variant="outlined" size="small" />
                                                                </Grid>
                                                            </Grid>
                                                        )
                                                    })}
                                                </CardContent>
                                            </Card> :
                                            null}
                                    </Box>
                                </Grid>
                            </Grid>
                        </Content>
                        <Sider style={{backgroundColor: '#f9f9f9'}} width={500}>
                            <Box p={1} ml={2}>
                                <Typography variant="h5" color="primary">
                                    Pattern Summary
                                </Typography>
                            </Box>
                            <Box ml={1} mr={1}>
                                <NestedList buggy_pattern={buggy_pattern} changeSelectPattern={this.changeSelectPattern} detectedSentimentalPattern={traceToPlot === '' ? sentimental_pattern.slice(0, 30) : possibleSentimentalPattern} hasTraceToPlot={traceToPlot !== ''}
                                            selectedIndex={this.state.patternSelectedIndex} setSelectedIndex={this.changePatternSelectedIndex} changeSearchText={this.changeSearchText}/>
                            </Box>
                        </Sider>
                    </Layout>
                    <Layout style={{ height: 500, backgroundColor: '#f9f9f9' }}>
                        <Box p={1} ml={2}>
                            <Typography variant="h5" color="primary">
                                Instance
                            </Typography>
                        </Box>
                        <Box p={1} ml={1} mr={1}>
                            <QuickFilteringGrid data={filteredRows} changeSelectTrace={this.changeSelectTrace} searchText={searchText} matchPattern={selectedPatterns.split('_')[0]}
                                                handleDatasetChange={this.handleDatasetChange} selectedDataset={this.state.dataset}/>
                        </Box>
                    </Layout>
                </Layout>
                <Footer style={{backgroundColor: '#f9f9f9'}}>
                    <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        flexWrap: 'wrap',
                    }}>
                        <Button
                            style={{color: "#007c41"}}
                            size="small"
                            startIcon={<CopyrightIcon />}
                            disabled={true}
                        >
                            University of Alberta
                        </Button>
                    </div>
                </Footer>
            </div>
        )
    }
}