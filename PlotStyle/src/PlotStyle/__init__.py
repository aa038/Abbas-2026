import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plotStyle():
    """Set up the plotting style to match the example."""
    plt.rcParams['xtick.color'] = "323034"
    plt.rcParams['ytick.color'] = "323034"
    plt.rcParams['text.color'] = "323034"
    plt.rcParams['lines.markeredgecolor'] = "black"
    plt.rcParams['patch.facecolor'] = "bc80bd"
    plt.rcParams['patch.force_edgecolor'] = True
    plt.rcParams['patch.linewidth'] = 0.8
    plt.rcParams['scatter.edgecolors'] = "black"
    plt.rcParams['grid.color'] = "b1afb5"
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.title_fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['font.size'] = 15
    plt.rcParams['axes.prop_cycle'] = "(cycler('color', ['1f77b4', 'fdb462', 'b3de69', 'fb8072', 'bc80bd', 'fccde5', '8dd3c7', 'ffed6f', 'bebada', '80b1d3', 'ccebc5', 'd9d9d9']))"
    plt.rcParams['mathtext.fontset'] = "stix"
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = ['Calibri']
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.fontsize'] = 13
    plt.rcParams['legend.edgecolor'] = "black"
    plt.rcParams['legend.borderpad'] = 0.2
    plt.rcParams['legend.columnspacing'] = 1.5
    plt.rcParams['legend.labelspacing'] = 0.4
    plt.rcParams['text.usetex'] = False
    plt.rcParams['axes.labelsize'] = 17
    plt.rcParams['axes.titlelocation'] = "center"
    plt.rcParams['axes.formatter.use_mathtext'] = True
    plt.rcParams['axes.autolimit_mode'] = "round_numbers"
    plt.rcParams['axes.labelpad'] = 3
    plt.rcParams['axes.formatter.limits'] = (-4, 4)
    plt.rcParams['axes.labelcolor'] = "black"
    plt.rcParams['axes.edgecolor'] = "black"
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['axes.grid'] = False
    plt.rcParams['axes.spines.right'] = True
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.top'] = True
    plt.rcParams['figure.titlesize'] = 18
    plt.rcParams['figure.dpi'] = 300

    plt.rcParams['xtick.major.size'] = 8
    plt.rcParams['ytick.major.size'] = 8
    plt.rcParams['xtick.minor.size'] = 4
    plt.rcParams['ytick.minor.size'] = 4

    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1