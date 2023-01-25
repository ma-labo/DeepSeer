import * as React from 'react';
import PropTypes from 'prop-types';
import Typography from '@material-ui/core/Typography';
import Paper from '@material-ui/core/Paper';
import Popper from '@material-ui/core/Popper';
import IconButton from '@material-ui/core/IconButton';
import TextField from '@material-ui/core/TextField';
import {DataGrid, GridToolbarDensitySelector, GridToolbarFilterButton} from '@material-ui/data-grid';
import ClearIcon from '@material-ui/icons/Clear';
import SearchIcon from '@material-ui/icons/Search';
import Button from '@material-ui/core/Button';
import BugReportIcon from '@material-ui/icons/BugReport';
import CheckCircleOutlineIcon from '@material-ui/icons/CheckCircleOutline';
import SentimentSatisfiedAltIcon from '@material-ui/icons/SentimentSatisfiedAlt';
import SentimentVeryDissatisfiedIcon from '@material-ui/icons/SentimentVeryDissatisfied';
import StorageIcon from '@material-ui/icons/Storage';
import PieChartIcon from '@material-ui/icons/PieChart';
import Box from '@material-ui/core/Box';
import Tooltip from '@material-ui/core/Tooltip';

import {createMuiTheme} from '@material-ui/core/styles';
import {makeStyles} from '@material-ui/styles';
import PermIdentityIcon from "@material-ui/icons/PermIdentity";
import ComputerIcon from "@material-ui/icons/Computer";

const defaultTheme = createMuiTheme();
const useStyles = makeStyles(
    (theme) => ({
        root: {
            padding: theme.spacing(0.5, 0.5, 0),
            justifyContent: 'space-between',
            display: 'flex',
            alignItems: 'flex-start',
            flexWrap: 'wrap',
        },
        textField: {
            [theme.breakpoints.down('xs')]: {
                width: '100%',
            },
            margin: theme.spacing(1, 0.5, 1.5),
            '& .MuiSvgIcon-root': {
                marginRight: theme.spacing(0.5),
            },
            '& .MuiInput-underline:before': {
                borderBottom: `1px solid ${theme.palette.divider}`,
            },
        },
    }),
    { defaultTheme },
);

function getRegexPattern(texts) {
    if (texts.length === 0) {
        return ''
    }
    if ((texts.slice(0, 3) === 're\'') && (texts.length > 3)) {
        return texts.slice(3, texts.length) + '[\\s]|' + texts.slice(3, texts.length) + '$'
    }
    else {
        let regexPattern = ''
        let splitTexts = texts.split(' ')
        while (splitTexts[0] === '') {
            splitTexts = splitTexts.slice(1, splitTexts.length)
        }
        regexPattern = splitTexts.join('+.*')
        return regexPattern
    }
}

const ColourfulText = React.memo(function ColourfulText(props) {
    const { value, seq_label, width, pl, hp} = props;
    const symbol = value.includes('➞') ? '➞' : ' '
    const items = value.split(symbol);
    return (
        <div>
            <Box pl={pl} pr={2}>
                <Typography variant="body2" display='inline' color={seq_label[0] === 0 ? 'primary' : 'secondary'} style={hp.has(0) ? {wordWrap: "break-word", backgroundColor: "yellow"} : {wordWrap: "break-word"}}>
                    {items.shift()}
                </Typography>
                {
                    items.map((item, index) => {
                        return (
                            <Box display='inline' maxWidth={width} minWidth={width}>
                                <Typography variant="body2" display='inline' style={{wordWrap: "break-word", color: "grey"}}>
                                    {symbol}
                                </Typography>
                                <Typography variant="body2" display='inline' color={seq_label.slice(1, seq_label.length)[index] === 0 ? 'primary' : 'secondary'} style={hp.has(index + 1) ? {wordWrap: "break-word", backgroundColor: "yellow"} : {wordWrap: "break-word"}}>
                                    {item.slice(0,2) === '##' ? item.slice(2, item.length) : item}
                                </Typography>
                            </Box>
                        )
                    })
                }
            </Box>
        </div>
    )
});

const ShortColourfulText = React.memo(function ShortColourfulText(props) {
    const { value, seq_label, width, pl, hp} = props;
    const symbol = value.includes('➞') ? '➞' : ' '
    const startPos = Array.from(hp)[0]
    const endPos = Array.from(hp)[hp.size - 1]
    let items;
    let label;
    let dynamic_hp;
    if (startPos < 15) {
        items = value.split(symbol).slice(0, 30);
        label = seq_label.slice(0, 30);
        dynamic_hp = hp
    }
    else if ((seq_label.length - endPos) < 15) {
        items = value.split(symbol).slice(seq_label.length - 31, seq_label.length - 1);
        label = seq_label.slice(seq_label.length - 31, seq_label.length - 1);
        dynamic_hp = new Set(Array.from(hp).map((x) => x - (seq_label.length - 26)))
    }
    else {
        items = value.split(symbol).slice(startPos - 16, startPos + 15)
        label = seq_label.slice(startPos - 16, startPos + 15)
        dynamic_hp = new Set(Array.from(hp).map((x) => x - (startPos - 16)))
    }
    return (
        <div>
            <Box pl={pl} pr={2}>
                <Typography variant="body2" display='inline' color={label[0] === 0 ? 'primary' : 'secondary'} style={dynamic_hp.has(0) ? {wordWrap: "break-word", backgroundColor: "yellow"} : {wordWrap: "break-word"}}>
                    {items.shift()}
                </Typography>
                {
                    items.map((item, index) => {
                        return (
                            <Box display='inline' maxWidth={width} minWidth={width}>
                                <Typography variant="body2" display='inline' style={{wordWrap: "break-word", color: "grey"}}>
                                    {symbol}
                                </Typography>
                                <Typography variant="body2" display='inline' color={label.slice(1, label.length)[index] === 0 ? 'primary' : 'secondary'} style={dynamic_hp.has(index + 1) ? {wordWrap: "break-word", backgroundColor: "yellow"} : {wordWrap: "break-word"}}>
                                    {item}
                                </Typography>
                            </Box>
                        )
                    })
                }
            </Box>
        </div>
    )
});

function getHighlightPosition(text, phrases) {
    let pos = new Set();
    if (phrases === '') {
        return pos;
    }
    if (!text.includes('➞')) {
        let items = text.split(' ')
        let words = phrases.split(' ')
        for (let i = 0; i < items.length - words.length + 1; i++) {
            let j = 0;
            while(items[i+j] === words[j] && j < words.length) {
                j += 1;
            }
            if (j === words.length) {
                for (let k = i; k < i + j; k++) {
                    pos.add(k);
                }
            }
        }
    }
    return pos;
}

const GridCellExpand = React.memo(function GridCellExpand(props) {
    const { width, value, seq_label, highlightPos } = props;
    const wrapper = React.useRef(null);
    const cellDiv = React.useRef(null);
    const cellValue = React.useRef(null);
    const [anchorEl, setAnchorEl] = React.useState(null);
    const classes = useStyles();
    const [showFullCell, setShowFullCell] = React.useState(false);
    const [showPopper, setShowPopper] = React.useState(false);

    const handleMouseEnter = () => {
        const isCurrentlyOverflown = cellValue.current.scrollWidth > width;
        setShowPopper(isCurrentlyOverflown);
        setAnchorEl(cellDiv.current);
        setShowFullCell(true);
    };

    const handleMouseLeave = () => {
        setShowFullCell(false);
    };

    React.useEffect(() => {
        if (!showFullCell) {
            return undefined;
        }

        function handleKeyDown(nativeEvent) {
            // IE11, Edge (prior to using Bink?) use 'Esc'
            if (nativeEvent.key === 'Escape' || nativeEvent.key === 'Esc') {
                setShowFullCell(false);
            }
        }

        document.addEventListener('keydown', handleKeyDown);

        return () => {
            document.removeEventListener('keydown', handleKeyDown);
        };
    }, [setShowFullCell, showFullCell]);

    return (
        <div
            ref={wrapper}
            className={classes.root}
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
        >
            <div
                ref={cellDiv}
                style={{
                    height: 0,
                    width,
                    display: 'block',
                    position: 'absolute',
                    top: 0,
                }}
            />
            <div ref={cellValue} className="cellValue">
                {/*{highlightPos.size > 0 ? <ShortColourfulText value={value} seq_label={seq_label} width={width} pl={0} hp={highlightPos}/> : <ColourfulText value={value} seq_label={seq_label} width={width} pl={0} hp={highlightPos}/>}*/}
                <ColourfulText value={value} seq_label={seq_label} width={width} pl={0} hp={highlightPos}/>
            </div>
            {showPopper && (
                <Popper
                    open={showFullCell && anchorEl !== null}
                    anchorEl={anchorEl}
                    style={{ width, marginLeft: -17 }}
                >
                    <Paper
                        elevation={1}
                        style={{ minHeight: wrapper.current.offsetHeight - 3 }}
                    >
                        {/*<Typography variant="body2" style={{ padding: 8, wordWrap: "break-word"}}>*/}
                        {/*    {value}*/}
                        {/*</Typography>*/}
                        <ColourfulText value={value} seq_label={seq_label} width={width} pl={2} hp={highlightPos}/>
                    </Paper>
                </Popper>
            )}
        </div>
    );
});

GridCellExpand.propTypes = {
    value: PropTypes.string.isRequired,
    width: PropTypes.number.isRequired,
};

function renderCellExpand(params, searchText, matchPattern) {
    const highlightPos = getHighlightPosition(params.getValue(params.id, 'text').join(' '), searchText)
    return (
        <GridCellExpand
            value={typeof params.value === 'number' ? params.value.toString() : (params.field === 'trace' ?
                    params.value.join('➞').toString() : params.value.join(' ').toString())}
            width={params.colDef.width}
            seq_label={params.getValue(params.id, 'seq_label')}
            highlightPos={highlightPos}
            // highlightPos={params.field === 'text' ? highlightPos : (params.value.slice(Array.from(highlightPos)[0], Array.from(highlightPos)[highlightPos.size - 1] + 1).join(' ') === matchPattern ? highlightPos : new Set())}
        />
    );
}

renderCellExpand.propTypes = {
    /**
     * The column of the row that the current cell belongs to.
     */
    colDef: PropTypes.any.isRequired,
    /**
     * The cell value, but if the column has valueGetter, use getValue.
     */
    value: PropTypes.oneOfType([
        PropTypes.instanceOf(Date),
        PropTypes.number,
        PropTypes.object,
        PropTypes.string,
        PropTypes.bool,
    ]),
};

function QuickSearchToolbar(props) {
    const classes = useStyles();

    return (
        <div className={classes.root}>
            <div>
                <GridToolbarFilterButton />
                <GridToolbarDensitySelector />
                <Button onClick={props.onDatasetChange} color='primary' startIcon={<StorageIcon />}>
                    {props.selectedDataset === 'train' ? <div><Box display='inline' fontWeight='fontWeightBold'>Train</Box>/Test</div> : <div>Train/<Box display='inline' fontWeight='fontWeightBold'>Test</Box></div>}
                </Button>
                <Button color='primary' startIcon={<PermIdentityIcon />}>
                    {<div><Box display='inline' fontWeight='fontWeightBold'>Human Label</Box>{' Sincere: ' + (props.pos_neg_label[1] - props.pos_neg_label[0]).toString() + ' / ' + 'Insincere: ' + props.pos_neg_label[0].toString()}</div>}
                </Button>
                <Button color='primary' startIcon={<ComputerIcon />}>
                    {<div><Box display='inline' fontWeight='fontWeightBold'>Prediction</Box>{' Sincere: ' + (props.pos_neg_pred[1] - props.pos_neg_pred[0]).toString() + ' / ' + 'Insincere: ' + props.pos_neg_pred[0].toString()}</div>}
                </Button>
            </div>
            <Tooltip title="Add re' for regex matching." placement="top">
                <TextField
                    variant="standard"
                    value={props.value}
                    onChange={props.onChange}
                    onKeyDown={(e) => {
                        if (e.key === "Enter") {
                           props.enterKeyPress();
                        }
                    }}
                    placeholder="Search…"
                    className={classes.textField}
                    InputProps={{
                        startAdornment: <SearchIcon color='primary' fontSize="small" />,
                        endAdornment: (
                            <IconButton
                                title="Clear"
                                aria-label="Clear"
                                size="small"
                                style={{ visibility: props.value ? 'visible' : 'hidden' }}
                                onClick={props.clearSearch}
                            >
                                <ClearIcon fontSize="small" />
                            </IconButton>
                        ),
                    }}
                />
            </Tooltip>
        </div>
    );
}

QuickSearchToolbar.propTypes = {
    clearSearch: PropTypes.func.isRequired,
    enterKeyPress: PropTypes.func.isRequired,
    onChange: PropTypes.func.isRequired,
    value: PropTypes.string.isRequired,
};

function getCorrectness(params) {
    return (params.getValue(params.id, 'pred') === params.getValue(params.id, 'label')).toString();
}

function getLabel(params) {
    return params.value == 0 ? 'Sincere' : 'Insincere'
}

export default function QuickFilteringGrid(data) {

    const [selectedRows, setSelectedRows] = React.useState();
    const [searchText, setSearchText] = React.useState('');

    const select_row = (thisRow) => {
        setSelectedRows(thisRow);
        select_trace(thisRow);
    }

    const select_trace = (trace) => {
        data.changeSelectTrace(trace);
    }

    const data_rows = data.data;

    const searchHighlightText = data.searchText === '' ? (searchText === '' ? '' : (searchText.slice(0, 3) === 're\'' ? searchText.slice(3, searchText.length) : '')) : data.searchText

    const data_columns = [
        { field: 'id', headerName: 'Index', width: 115 },
        { field: 'trace',
            headerName: 'Trace',
            flex: 1,
            minWidth: 420,
            renderCell: (params) => renderCellExpand(params, searchHighlightText, data.matchPattern),
        },
        { field: 'text',
            headerName: 'Text',
            flex: 1,
            minWidth: 420,
            renderCell: (params) => renderCellExpand(params, searchHighlightText, data.matchPattern),
        },
        { field: 'pred',
            headerName: 'Prediction',
            width: 165,
            valueGetter: getLabel,
            renderCell: (params) => (
                <Button
                    variant="outlined"
                    style={{maxWidth: '110px', minWidth: '110px',
                        borderColor: (params.value === 'Sincere' ? '#2196f3' : '#b26500'),
                        color: (params.value === 'Sincere' ? '#2196f3' : '#b26500')}}
                    size="small"
                    endIcon={params.value === 'Sincere' ? <SentimentSatisfiedAltIcon /> : <SentimentVeryDissatisfiedIcon /> }
                    disabled={true}
                >
                    {params.value}
                </Button>
            ), },
        { field: 'label',
            headerName: 'Human Label',
            width: 165,
            valueGetter: getLabel,
            renderCell: (params) => (
                <Button
                    variant="outlined"
                    style={{maxWidth: '110px', minWidth: '110px',
                        borderColor: (params.value === 'Sincere' ? '#2196f3' : '#b26500'),
                        color: (params.value === 'Sincere' ? '#2196f3' : '#b26500')}}
                    size="small"
                    endIcon={params.value === 'Sincere' ? <SentimentSatisfiedAltIcon /> : <SentimentVeryDissatisfiedIcon /> }
                    disabled={true}
                >
                    {params.value}
                </Button>
            ), },
        { field: 'correctness',
            headerName: 'Correctness',
            width: 160,
            valueGetter: getCorrectness,
            renderCell: (params) => (
                <Button
                    variant="outlined"
                    style={{maxWidth: '80px', minWidth: '80px',
                        borderColor: (params.value === 'true' ? '#00a152' : '#ff1744'),
                        color: (params.value === 'true' ? '#00a152' : '#ff1744')}}
                    size="small"
                    endIcon={params.value === 'true' ? <CheckCircleOutlineIcon /> : <BugReportIcon /> }
                    disabled={true}
                >
                    {params.value}
                </Button>
            ), },
    ];

    const [rows, setRows] = React.useState(data_rows);
    const [pageSize, setPageSize] = React.useState(5);

    const handlePageSizeChange = (params) => {
        setPageSize(params.pageSize);
    };

    const requestSearch = (searchValue) => {
        const searchRegex = new RegExp(getRegexPattern(searchValue), 'i');
        const filteredRows = data_rows.filter((row) => {
            return searchRegex.test(row['text'].join(' ').toString());
        });
        setRows(filteredRows);
    };

    React.useEffect(() => {
        setRows(data_rows);
    }, [data_rows]);

    return (
        <div style={{ height: 420, width: '100%' }}>
            <DataGrid
                onSelectionModelChange={newSelection=>{
                    select_row(newSelection)
                }}
                components={{ Toolbar: QuickSearchToolbar }}
                localeText={{
                    footerRowSelected: (count) => `1 row selected, hold ctrl (command) and click on it to unselect.`
                }}
                rows={rows}
                columns={data_columns}
                pageSize={pageSize}
                onPageSizeChange={handlePageSizeChange}
                rowsPerPageOptions={[5, 10, 20]}
                pagination
                componentsProps={{
                    toolbar: {
                        value: data.searchText === '' ? searchText : 're\'' + data.searchText,
                        onChange: (event) => setSearchText(event.target.value),
                        enterKeyPress: () => requestSearch(searchText),
                        clearSearch: () => {setSearchText(''); requestSearch('')},
                        onDatasetChange: () => data.handleDatasetChange(),
                        selectedDataset: data.selectedDataset,
                        pos_neg_pred: [rows.reduce((a, b) => a + (b['pred'] || 0), 0), rows.length],
                        pos_neg_label: [rows.reduce((a, b) => a + (b['label'] || 0), 0), rows.length],
                    },
                }}
            />
        </div>
    );
}