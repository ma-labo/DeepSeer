"""
Name   : state_graph.py
Author : Zhijie Wang
Time   : 2021/7/7
"""

from graphviz import Digraph

from graphics.color_utils import get_color_hex, tint_color


class StateGraph(object):
    def __init__(self, state_word, word_state, transition, state_labels, base_node_size=2.):
        """

        :param state_word: {1:{'a':{...}, 'b':{...}}, 2:{...}}
        :param word_state: {'a':{1:{...}, 2:{...}}, 'b':{...}}
        :param transition: {'a':{1:{...}, 2:{...}}, 'b':{...}}
        :param state_labels: {1:{1:20, 0: 11}, 2:{...}, ...}
        """
        self.state_word = state_word
        self.word_state = word_state
        self.state_labels = state_labels
        self.state_freq = {}
        self.states = [i for i in self.state_word.keys()]
        self.colors = get_color_hex(self.states, True, "Spectral")
        self.state_property = []
        for key in self.state_word.keys():
            word_freq = len(self.state_word[key].keys())
            state_freq = sum([len(self.state_word[key][i]) for i in self.state_word[key].keys()])
            self.state_freq[key] = state_freq
            self.state_property.append((key, state_freq, word_freq))
        self.state_property.sort(key=lambda x: x[1], reverse=True)
        self.base_node_size = base_node_size
        self.node_size = {}
        for (state, state_freq, word_freq) in self.state_property:
            self.node_size[state] = round(base_node_size * (state_freq / self.state_property[0][1]), 1)
        self.edges = {}
        self.most_popular_trace = 0
        self.topk_edges = {state: {'in': {}, 'out': {}} for state in self.states}
        for tran in transition:
            if (tran[0], tran[2]) not in self.edges.keys():
                self.edges[(tran[0], tran[2])] = 1
            else:
                self.edges[(tran[0], tran[2])] += 1
                if self.edges[(tran[0], tran[2])] > self.most_popular_trace:
                    self.most_popular_trace = self.edges[(tran[0], tran[2])]
            if tran[0] != tran[2]:
                if tran[0] not in self.topk_edges[tran[2]]['in'].keys():
                    self.topk_edges[tran[2]]['in'][tran[0]] = self.edges[(tran[0], tran[2])]
                else:
                    self.topk_edges[tran[2]]['in'][tran[0]] = max(self.topk_edges[tran[2]]['in'][tran[0]],
                                                                  self.edges[(tran[0], tran[2])])
                if tran[2] not in self.topk_edges[tran[0]]['out'].keys():
                    self.topk_edges[tran[0]]['out'][tran[2]] = self.edges[(tran[0], tran[2])]
                else:
                    self.topk_edges[tran[0]]['out'][tran[2]] = max(self.topk_edges[tran[0]]['out'][tran[2]],
                                                                   self.edges[(tran[0], tran[2])])
        for key in self.topk_edges.keys():
            self.topk_edges[key]['in'] = sorted(self.topk_edges[key]['in'].items(), key=lambda item: item[1],
                                                reverse=True)
            self.topk_edges[key]['out'] = sorted(self.topk_edges[key]['out'].items(), key=lambda item: item[1],
                                                 reverse=True)
        self.sorted_edges = sorted(self.edges.items(), key=lambda item: item[1], reverse=True)
        self.sorted_edges_wo_self_incident = [v for v in self.sorted_edges if v[0][0] != v[0][1]]

    def draw_states(self, base_node_size=2.):
        """
        Draw all states
        :param base_node_size: size of the largest node
        :return: Digraph object
        """
        dot = Digraph(comment="DeepStellar")
        dot.attr(dpi='150.0')
        for (state, state_freq, word_freq) in self.state_property:
            node_size = round(base_node_size * (state_freq / self.state_property[0][1]), 1)
            dot.node(str(state), 'S%d' % state, shape="circle", width=str(node_size),
                     color=self.colors[state][0], style='filled', fillcolor=self.colors[state][1])

        return dot

    def draw_state_transition(self, topk=2):
        """
        Draw all states and transitions between them
        :param transition: [(prev_state, curr_embedding, curr_state), ...]
        :param base_node_size: size of largest node
        :param topk: how many incident edges to be plotted of each node.
        :return: Digraph object
        """
        dot = Digraph(comment="DeepStellarTransition")
        dot.attr(dpi='300.0', ratio="fill", size="11.7,8.3!", margin='0')
        dot.node(str(0), 'S0', shape="circle", color=self.colors[0][0], style='filled', fillcolor=self.colors[0][1])
        for (state, state_freq, word_freq) in self.state_property:
            dot.node(str(state), 'S%d' % state, shape="circle", width=str(self.node_size[state]),
                     color=self.colors[state][0], style='filled', fillcolor=self.colors[state][1])
        edge_list = set()
        for state in self.states:
            # in edge
            in_edge_num = len(self.topk_edges[state]['in'])
            for i in range(min(in_edge_num, topk)):
                edge_width = (self.edges[(self.topk_edges[state]['in'][i][0], state)] / self.most_popular_trace) * 9. + 1.
                if (str(self.topk_edges[state]['in'][i][0]), str(state)) not in edge_list:
                    dot.edge(str(self.topk_edges[state]['in'][i][0]), str(state), penwidth=str(edge_width))
                    edge_list.add((str(self.topk_edges[state]['in'][i][0]), str(state)))
            # out edge
            out_edge_num = len(self.topk_edges[state]['out'])
            for i in range(min(out_edge_num, topk)):
                edge_width = (self.edges[(state, self.topk_edges[state]['out'][i][0])] / self.most_popular_trace) * 9. + 1.
                if (str(state), str(self.topk_edges[state]['out'][i][0])) not in edge_list:
                    dot.edge(str(state), str(self.topk_edges[state]['out'][i][0]), penwidth=str(edge_width))
                    edge_list.add((str(state), str(self.topk_edges[state]['out'][i][0])))
        # for edge in edges.keys():
        #     edge_width = (edges[(edge[0], edge[1])] / most_popular_trace) * 9. + 1.
        #     dot.edge(str(edge[0]), str(edge[1]), penwidth=str(edge_width))
        return dot

    def draw_single_transition(self, trace, text):
        """
        Draw transition of a single trace
        :param trace: [1,2,3,4, ...]
        :param text: ['a','b','c','d', ...]
        :return: Digraph object
        """
        dot = Digraph(comment="A trace")
        dot.attr(dpi='300.0', ratio="fill", size="11.7,8.3!", margin='0')
        state_text = {}
        for state in set(trace):
            state_text[state] = 'S%d' % state
        for state in set(trace):
            dot.node(str(state), state_text[state], shape="circle", width=str(self.node_size[state]),
                     color=self.colors[state][0], style='filled', fillcolor=self.colors[state][1])
        dot.node(str(0), 'S0', shape="circle", color=self.colors[0][0], style='filled', fillcolor=self.colors[0][1])
        dot.edge(str(0), str(trace[0]), text[0])
        for i in range(1, len(trace)):
            dot.edge(str(trace[i - 1]), str(trace[i]), text[i])
        dot.attr(label=' '.join(text))

        return dot

    def draw_single_transition_w_labels(self, trace, text, seq_labels):
        """
        Draw transition of a single trace
        :param trace: [1,2,3,4, ...]
        :param text: ['a','b','c','d', ...]
        :param seq_labels: [0, 0, 0, 1, ...]
        :return: Digraph object
        """
        dot = Digraph(comment="A trace")
        dot.attr(dpi='300.0', ratio="fill", size="11.7,8.3!", margin='0')
        text = [v.replace('\\', '/') for v in text]
        state_text = {}
        for state in set(trace):
            state_text[state] = 'S%d' % state
        n_classes = 0
        for state in set(trace):
            tot = [self.state_labels[state][key] for key in sorted(self.state_labels[state].keys())]
            n_classes = len(tot)
            rate = [str(round(v / sum(tot), 2)) for v in tot]
            rate = ' '.join(rate)
            dot.node(str(state), state_text[state] + '\n' + rate, shape="circle",
                     width=str(self.node_size[state]), color=self.colors[state][0], style='filled',
                     fillcolor=self.colors[state][1])
        dot.node(str(0), 'S0', shape="circle", color=self.colors[0][0], style='filled', fillcolor=self.colors[0][1])

        def get_edge_color(label1, label2=None, n_classes=2):
            if n_classes == 2:
                color_str = '/paired9/'
                if not label2 or label1 == label2:
                    return color_str + str(4 * label1 + 2)
                else:
                    return color_str + str(4 * label1 + 2) + ';0.5:' + color_str + str(4 * label2 + 2)
            else:
                color_str = '/paired' + str(n_classes) + '/'
                if not label2 or label1 == label2:
                    return color_str + str(label1 + 1)
                else:
                    return color_str + str(label1 + 1) + ';0.5:' + color_str + str(label2 + 1)

        dot.edge(str(0), str(trace[0]), text[0], color=get_edge_color(label1=seq_labels[0][0], label2=None,
                                                                      n_classes=n_classes))
        for i in range(1, len(trace)):
            dot.edge(str(trace[i - 1]), str(trace[i]), text[i],
                     color=get_edge_color(seq_labels[i - 1][0], seq_labels[i][0], n_classes))
        dot.attr(label=' '.join(text))

        return dot

    def draw_word_stat(self, word):
        """
        Draw analysis of a single word
        :param word: 'abcd'
        :param base_node_size: size of largest node
        :return: Digraph object
        """
        dot = Digraph(comment="Word statistic")
        dot.attr(dpi='150.0')
        state_property = []
        for state in self.word_state[word].keys():
            state_freq = self.word_state[word][state]['freq']
            state_property.append((state, state_freq))
        state_property.sort(key=lambda x: x[1], reverse=True)
        for (state, state_freq) in state_property:
            dot.node(str(state), 'S%d' % state, shape="circle", width=str(self.node_size[state]),
                     color=self.colors[state][0], style='filled', fillcolor=self.colors[state][1])
        dot.attr(label='Statistics of \'%s\'' % word)

        return dot

    def draw_sequence(self, trace, text):
        """
        Draw a sequence
        :param trace: [1,2,3,4, ...]
        :param text: ['a','b','c','d', ...]
        :return: Digraph object
        """
        dot = Digraph(comment="A sequence")
        dot.attr(dpi='150.0')
        text = [v.replace('\\', '/') for v in text]
        dot.node('0', 'S%d\n%s' % (trace[0], text[0]),
                 color=self.colors[trace[0]][0], style='filled', fillcolor=self.colors[trace[0]][1])
        for i in range(1, len(trace)):
            dot.node(str(i), 'S%d\n%s' % (trace[i], text[i]),
                     color=self.colors[trace[i]][0], style='filled', fillcolor=self.colors[trace[i]][1])
            dot.edge(str(i - 1), str(i))
        dot.attr(label=' '.join(text))

        return dot
