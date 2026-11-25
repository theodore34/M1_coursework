MPL_STYLE = {
    "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "font.size": 12, "errorbar.capsize": 2, "legend.fontsize": 12,
    "figure.autolayout": True,
}
colors = ["#2B5C9E",   # bleu ardoise
                "#D9891A",   # ocre
                "#0FA18A",   # émeraude
                "#9C3D7C"]   # prune


def print_results():
    """
    Print every results cleanly
    """
    for key, value in results.items():
        if isinstance(value, (float, np.floating)):
            value = f"{value:.3g}"
        print(f"row {key}: {value}")
