from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(r"D:\Github\Personal-Project\textual data analysis")
OUTPUT_DIR = PROJECT_ROOT / "resources" / "figures" / "methods"


def make_points(rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "Win": rng.normal(loc=(-1.95, -1.05), scale=(0.32, 0.28), size=(42, 2)),
        "Mixed": rng.normal(loc=(0.05, 1.65), scale=(0.36, 0.32), size=(42, 2)),
        "Lose": rng.normal(loc=(1.95, -0.25), scale=(0.34, 0.28), size=(42, 2)),
    }


def plot_boundary(ax, point_a, point_b, color="#64748b") -> None:
    x0, y0 = point_a
    x1, y1 = point_b
    ax.plot([x0, x1], [y0, y1], color=color, linewidth=1.6)

    dx = x1 - x0
    dy = y1 - y0
    norm = np.hypot(dx, dy)
    nx = -dy / norm
    ny = dx / norm
    margin = 0.18
    for sign in (-1, 1):
        ax.plot(
            [x0 + sign * margin * nx, x1 + sign * margin * nx],
            [y0 + sign * margin * ny, y1 + sign * margin * ny],
            color=color,
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            alpha=0.55,
        )


def main() -> None:
    rng = np.random.default_rng(42)
    points = make_points(rng)
    colors = {"Win": "#2f9e44", "Mixed": "#f97316", "Lose": "#2563eb"}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    ax.set_facecolor("#f8fafc")

    for label, xy in points.items():
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=34,
            color=colors[label],
            edgecolor="white",
            linewidth=0.65,
            alpha=0.9,
            label=label,
            zorder=3,
        )

    support_vectors = np.vstack(
        [
            points["Win"][np.argsort(points["Win"][:, 0] + points["Win"][:, 1])[-2:]],
            points["Mixed"][np.argsort(points["Mixed"][:, 0] - points["Mixed"][:, 1])[-2:]],
            points["Lose"][np.argsort(-points["Lose"][:, 0] + points["Lose"][:, 1])[-2:]],
        ]
    )
    ax.scatter(
        support_vectors[:, 0],
        support_vectors[:, 1],
        s=130,
        facecolors="none",
        edgecolors="#111827",
        linewidths=1.3,
        zorder=4,
        label="Margin-defining points (conceptual)",
    )

    plot_boundary(ax, (-2.85, 0.40), (0.55, -2.35))
    plot_boundary(ax, (-0.98, -2.35), (1.10, 2.65))
    plot_boundary(ax, (-0.92, 0.55), (2.90, 1.23))

    ax.text(-2.15, -1.55, "Win", color=colors["Win"], fontsize=12, fontweight="bold")
    ax.text(-0.12, 2.13, "Mixed", color=colors["Mixed"], fontsize=12, fontweight="bold")
    ax.text(2.12, -0.78, "Lose", color=colors["Lose"], fontsize=12, fontweight="bold")
    ax.text(
        1.55,
        1.35,
        "Decision boundaries",
        color="#475569",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    ax.text(
        -2.75,
        2.45,
        "Dashed lines indicate margins",
        color="#475569",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )

    ax.set_title("SVM Classification Schematic", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Feature dimension 1", fontsize=11)
    ax.set_ylabel("Feature dimension 2", fontsize=11)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2.6, 2.8)
    ax.grid(True, color="#cbd5e1", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=9)

    for suffix in ("png", "pdf", "svg"):
        out_path = OUTPUT_DIR / f"svm_classification_schematic.{suffix}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
