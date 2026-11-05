from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_unit_hydrograph(response: np.ndarray, path: str | Path, title: str = "Derived Unit Hydrograph") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    response = np.asarray(response)
    peak = int(np.argmax(response))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(response, color="#1f77b4", marker="o", markersize=4, label="Unit hydrograph")
    ax.fill_between(np.arange(len(response)), response, 0, color="#1f77b4", alpha=0.15)
    ax.plot(peak, response[peak], "o", color="#d62728", markersize=8)
    ax.set_xlabel("Lag time (days)")
    ax.set_ylabel("Response")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_kernels(kernels: np.ndarray, path: str | Path, max_kernels: int = 4) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kernels = np.asarray(kernels)
    n_plot = min(max_kernels, len(kernels))

    fig, axes = plt.subplots(1, n_plot, figsize=(3 * n_plot, 3), squeeze=False)
    for idx in range(n_plot):
        ax = axes[0, idx]
        kernel = kernels[idx]
        peak = int(np.argmax(kernel))
        ax.plot(kernel, color="#1f77b4", linewidth=2)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.6)
        ax.plot(peak, kernel[peak], "o", color="#d62728", markersize=5)
        ax.set_title(f"Kernel {idx + 1}")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

