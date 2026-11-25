"""Figure helpers.  PROVIDED — do not edit."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY = "#21295C"; DEEP = "#065A82"; TEAL = "#1C7293"
ORANGE = "#eb6834"; BLUE = "#2a78d6"; GREEN = "#1baf7a"
MUTED = "#5A6472"; GRID = "#d9d9d6"
SERIES = [BLUE, ORANGE, GREEN, TEAL]


def style(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=11, loc="left", pad=8, color="#1A1A1A")
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    return ax


def save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  figure -> {path}")
