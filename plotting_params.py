## plotting parameters ##
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

font_size = 18

my_rcparams = {
    #"text.usetex": True,
    "font.family": "serif",
    "font.size": font_size,

    "axes.labelsize": font_size,
    "axes.titlesize": font_size * 1.2,
    "legend.fontsize": font_size * 0.8,
    "xtick.labelsize": font_size * 0.9,
    "ytick.labelsize": font_size * 0.9,

    "xtick.bottom": True,
    "xtick.labelbottom": True,
    "xtick.top": False,
    "xtick.labeltop": False,

    "xtick.minor.visible": True,
    "ytick.minor.visible": True,

    "ytick.right": False,
    "ytick.labelright": False,
    "ytick.left": True,
    "ytick.labelleft": True,

    "xtick.direction": "in",
    "ytick.direction": "in",

    "xtick.major.size": 5,
    "xtick.minor.size": 3,
    "xtick.major.width": 1,
    "xtick.minor.width": 1,
    "ytick.major.size": 5,
    "ytick.minor.size": 3,
    "ytick.major.width": 1,
    "ytick.minor.width": 1,

    "legend.frameon": False,
    "legend.loc": "best",
    "legend.shadow": True,
    "legend.framealpha": 1,

    "axes.linewidth": 1,
    "axes.labelcolor": "k",
    "axes.edgecolor": "k",
    "axes.grid": False,
    #"axes.grid.which": "both",
    "axes.xmargin": 0,
    "axes.ymargin": 0,
    # "axes.prop_cycle": plt.cycler('color', ["#FF7698", '#858AE3', '#FFB20F', '#2A1E5C',  "#DF5431", '#582707', '#5BC0EB', "#754492", "#740909", '#305252']),
    "axes.prop_cycle": plt.cycler('color', ["#EB3A34", '#539691', '#6B574D', '#3496EB',  "#537796", '#EA7034', '#34EBDE', "#413A39", "#AB6E50", '#965653']),

    "xtick.color": "k",
    "ytick.color": "k",

    "xtick.major.pad": 8,
    "ytick.major.pad": 8,
    "axes.labelpad": 8,

    "hatch.linewidth": 3,
    "hatch.color": "w",

    "grid.color": "gray",
    "grid.alpha": 0.5,

    "axes.facecolor": "w",
}

def use_my_style():
    plt.rcParams.update(my_rcparams)

