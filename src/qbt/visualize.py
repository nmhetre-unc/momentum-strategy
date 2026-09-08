"""
Equity curve and drawdown charts, saved as PNG files rather than popped
up in an interactive window (this is meant to run from the command line).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from qbt.analytics import drawdown_series

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EQUITY_PATH = str(REPO_ROOT / "equity_curve.png")
DEFAULT_DRAWDOWN_PATH = str(REPO_ROOT / "drawdown.png")


def plot_equity_curve(result: pd.DataFrame, title: str, out_path: str = DEFAULT_EQUITY_PATH):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result.index, result["equity_curve"], label="Strategy")
    ax.plot(result.index, result["benchmark_curve"], label="Buy & Hold", linestyle="--")
    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_drawdown(result: pd.DataFrame, title: str, out_path: str = DEFAULT_DRAWDOWN_PATH):
    drawdown = drawdown_series(result["equity_curve"])

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(result.index, drawdown, 0, color="crimson", alpha=0.4)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
