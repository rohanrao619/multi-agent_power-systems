import matplotlib.pyplot as plt

# Common plot style for all plots in the thesis
def set_plot_style():
    
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "font.family": "serif",       # use serif fonts
        "mathtext.fontset": "cm",     # Computer Modern math
        "mathtext.rm": "serif",
        "axes.labelsize": 20,
        "axes.labelpad": 8,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 18,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

aid_mapping = {'consumer_1': '69',
               'prosumer_1': '110',
               'prosumer_2': '119',
               'consumer_2': '184',
               'consumer_3': '189',
               'prosumer_3': '193',
               'consumer_4': '212',
               'prosumer_4': '256'}

color_mapping = {
    "prosumer_1": "#8B4513",  # Brown
    "prosumer_2": "#FFD900",  # Yellow
    "prosumer_3": "#FF8C00",  # Orange
    "prosumer_4": "#1cbd00",  # Green
    "consumer_1": "#FF1010",  # Red
    "consumer_2": "#ff1aff",  # Pink
    "consumer_3": "#1b1bff",  # Blue
    "consumer_4": "#880088",  # Purple
    "total": "#525252"        # Grey
}