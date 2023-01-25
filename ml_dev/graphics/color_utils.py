"""
Name   : color_utils.py
Author : Zhijie Wang
Time   : 2021/7/8
"""

import seaborn as sns
from matplotlib.colors import rgb2hex, hex2color


def create_color_map(*args):
    return sns.color_palette(*args, as_cmap=True)


def get_color_hex(states, add_init_state=True, *args):
    cmap = create_color_map(*args)
    colors = {}
    if add_init_state:
        states.append(0)
    for state in states:
        color_value = (state - min(states)) / (max(states) - min(states))
        color_value = color_value * 0.6 + 0.2
        edge_color = cmap(color_value)
        bg_color = (edge_color[0] + (1 - edge_color[0]) * 0.75,
                    edge_color[1] + (1 - edge_color[1]) * 0.75,
                    edge_color[2] + (1 - edge_color[2]) * 0.75, edge_color[3])
        colors[state] = (rgb2hex(edge_color), rgb2hex(bg_color))
    return colors


def tint_color(color_hex, rate=0.75):
    color = hex2color(color_hex)
    tinted_color = (color[0] + (1 - color[0]) * rate,
                    color[1] + (1 - color[1]) * rate,
                    color[2] + (1 - color[2]) * rate)
    return rgb2hex(tinted_color)
