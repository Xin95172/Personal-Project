from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_DIRECT_SVM_HEATMAP_CONFIGS = [
    {
        "dataset": "Administrative",
        "feature": "BoW",
        "path": PROJECT_ROOT
        / "artifacts/reports/administrative_win_lose_mixed/no_leakage/bow/step3_runs/full_20260504_154912/direct_svm_chi2_k_svm_validation_grid.csv",
    },
    {
        "dataset": "Civil",
        "feature": "TF-IDF",
        "path": PROJECT_ROOT
        / "artifacts/reports/civil_win_lose_mixed/no_leakage/tfidf/step3_runs/full_20260504_154912/direct_svm_chi2_k_svm_validation_grid.csv",
    },
    {
        "dataset": "Criminal",
        "feature": "TF",
        "path": PROJECT_ROOT
        / "artifacts/reports/criminal_win_lose_mixed/no_leakage/tf/step3_runs/full_20260504_154912/direct_svm_chi2_k_svm_validation_grid.csv",
    },
    {
        "dataset": "Crim. w/ Civil",
        "feature": "BoW",
        "path": PROJECT_ROOT
        / "artifacts/reports/cwc_win_lose_mixed/no_leakage/bow/step3_runs/full_20260504_154912/direct_svm_chi2_k_svm_validation_grid.csv",
    },
]


def configure_chinese_matplotlib_fonts():
    """Configure matplotlib fonts for mixed Chinese and English labels."""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft JhengHei",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_direct_svm_validation_grids(configs=None):
    """Load Direct SVM K x C validation grids from existing CSV artifacts.

    Parameters
    ----------
    configs : list[dict] or None
        Each config needs ``dataset``, ``feature``, and ``path``.

    Returns
    -------
    list[dict]
        Each item contains metadata, original rows, pivot table, and best row.
    """
    configs = configs or DEFAULT_DIRECT_SVM_HEATMAP_CONFIGS
    grids = []

    for config in configs:
        path = Path(config["path"])
        if not path.exists():
            raise FileNotFoundError(f"Validation grid not found: {path}")

        df = pd.read_csv(path)
        required = {"K", "C", "Validation Macro F1"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")

        df = df.copy()
        df["K"] = df["K"].astype(int)
        df["C"] = df["C"].astype(float)
        df["Validation Macro F1"] = df["Validation Macro F1"].astype(float)

        pivot = (
            df.pivot_table(
                index="K",
                columns="C",
                values="Validation Macro F1",
                aggfunc="max",
            )
            .sort_index()
            .reindex(sorted(df["C"].unique()), axis=1)
        )
        best = df.loc[df["Validation Macro F1"].idxmax()].copy()

        grids.append(
            {
                "dataset": config["dataset"],
                "feature": config["feature"],
                "path": path,
                "data": df,
                "pivot": pivot,
                "best": best,
            }
        )

    return grids


def plot_direct_svm_k_c_validation_heatmap(
    configs=None,
    output_path=None,
    title="Direct SVM K x C Validation Macro F1",
    cmap="YlGnBu",
    figsize=(12, 7.2),
    dpi=300,
    annotate=True,
    mark_best=True,
    shared_color_scale=True,
    show=True,
):
    """Plot Direct SVM validation Macro F1 heatmaps over K and C.

    This function only reads existing validation-grid CSV files. It does not
    rerun feature selection or model training.

    Returns
    -------
    fig, axes, grids
        ``grids`` includes each dataset's loaded dataframe, pivot table, and
        best validation row.
    """
    configure_chinese_matplotlib_fonts()
    grids = load_direct_svm_validation_grids(configs)

    values = pd.concat([g["data"]["Validation Macro F1"] for g in grids])
    vmin = values.min() if shared_color_scale else None
    vmax = values.max() if shared_color_scale else None

    n = len(grids)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, constrained_layout=True)

    last_im = None
    for ax, grid in zip(axes.ravel(), grids):
        pivot = grid["pivot"]
        last_im = ax.imshow(
            pivot.values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )

        ax.set_title(f'{grid["dataset"]} ({grid["feature"]})', fontweight="bold")
        ax.set_xlabel("SVM C")
        ax.set_ylabel("K")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:g}" for c in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(k) for k in pivot.index])

        if annotate:
            import matplotlib.colors as mcolors
            norm = mcolors.Normalize(vmin=vmin or pivot.min().min(),
                                     vmax=vmax or pivot.max().max())
            cmap_fn = plt.get_cmap(cmap)
            for i, k in enumerate(pivot.index):
                for j, c in enumerate(pivot.columns):
                    value = pivot.loc[k, c]
                    r, g, b, _ = cmap_fn(norm(value))
                    # 感知亮度公式 (ITU-R BT.601)
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    color = "white" if lum < 0.45 else "black"
                    ax.text(
                        j, i, f"{value:.3f}",
                        ha="center", va="center",
                        fontsize=8, color=color,
                    )

        if mark_best:
            best = grid["best"]
            best_k = int(best["K"])
            best_c = float(best["C"])
            i = list(pivot.index).index(best_k)
            j = list(pivot.columns).index(best_c)
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="red",
                    linewidth=2.2,
                )
            )
            ax.text(j + 0.35, i - 0.35, "*", color="red", fontsize=14, fontweight="bold")

    for ax in axes.ravel()[len(grids) :]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.9, pad=0.02)
        cbar.set_label("Validation Macro F1")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes, grids
