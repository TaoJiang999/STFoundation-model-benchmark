"""
Clustering evaluation results table (scIB style)
- No column grouping (single-level header)
- Overall Score = directly from 'mean value' column in df
- Column order: AMI, ARI, FMI, Homogeneity, NMI, V_Measure, Overall
"""

from __future__ import annotations

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# 列配置
# ──────────────────────────────────────────────

COLUMN_ORDER = ["AMI", "ARI", "FMI", "HS", "NMI", "VM", "Overall"]

DISPLAY_NAMES: dict[str, str] = {
    "AMI":         "AMI",
    "ARI":         "ARI",
    "FMI":         "FMI",
    "HS":          "Homogeneity",
    "NMI":         "NMI",
    "VM":          "V-Measure",
    "Overall":     "Overall\nScore",
}


# ──────────────────────────────────────────────
# 主绘图函数
# ──────────────────────────────────────────────

def plot_clustering_eval(
    df: pd.DataFrame,
    mean_value_col: str = "mean value",
    save_dir: str | None = None,
    show: bool = True,
    figsize: tuple[float, float] | None = None,
    save_fig_name:str="cluster_metric",
) -> "Table":
    """
    绘制聚类评估结果表格（scIB 风格，无分组）。

    Parameters
    ----------
    df             : 行=方法名，列=指标名，须包含 'mean value' 列
    mean_value_col : Overall Score 来源列名，默认 'mean value'
    save_dir       : 保存目录，None 则不保存
    show           : 是否显示图像
    figsize        : 图像尺寸，None 自动计算
    """
    from plottable import ColumnDefinition, Table
    from plottable.cmap import normed_cmap
    from plottable.plots import bar

    plot_df = df.copy()

    # 将 'mean value' 映射为 'Overall'
    if mean_value_col in plot_df.columns:
        plot_df = plot_df.rename(columns={mean_value_col: "Overall"})
    else:
        raise KeyError(f"列 '{mean_value_col}' 不存在，请检查 df 列名。")

    # 按 Overall 降序排列
    # plot_df = plot_df.sort_values("Overall", ascending=False)

    # 只保留目标列，按定义顺序排列
    ordered_cols = [c for c in COLUMN_ORDER if c in plot_df.columns]
    plot_df = plot_df[ordered_cols].reset_index()
    plot_df = plot_df.rename(columns={plot_df.columns[0]: "Method"})

    # 数值列强制 float
    for col in plot_df.columns:
        if col != "Method":
            plot_df[col] = plot_df[col].astype(float)

    # ── colormap ──────────────────────────────
    def circle_cmap(col_data: pd.Series):
        return normed_cmap(col_data.astype(float), cmap=mpl.cm.PRGn, num_stds=2.5)

    # ── 列定义 ────────────────────────────────
    col_defs = [
        ColumnDefinition(
            "Method",
            width=1.5,
            textprops={"ha": "left", "weight": "bold"},
        )
    ]

    for col in ordered_cols:
        display = DISPLAY_NAMES.get(col, col)

        if col == "Overall":
            col_defs.append(
                ColumnDefinition(
                    col,
                    width=1.2,
                    title=display,
                    plot_fn=bar,
                    plot_kw={
                        "cmap": mpl.cm.YlGnBu,
                        "plot_bg_bar": False,
                        "annotate": True,
                        "height": 0.9,
                        "formatter": "{:.3f}",
                    },
                    border="left",
                )
            )
        else:
            col_defs.append(
                ColumnDefinition(
                    col,
                    title=display,
                    width=0.85,
                    textprops={
                        "ha": "center",
                        "bbox": {"boxstyle": "circle", "pad": 0.25},
                    },
                    cmap=circle_cmap(plot_df[col]),
                    formatter="{:.3f}",
                )
            )

    # ── 图像尺寸 ──────────────────────────────
    if figsize is None:
        figsize = (max(10, len(ordered_cols) * 1.0), 2.5 + 0.55 * len(plot_df))

    # ── 绘图 ──────────────────────────────────
    with mpl.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=figsize)
        # ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

        tab = Table(
            plot_df,
            cell_kw={"linewidth": 0, "edgecolor": "k"},
            column_definitions=col_defs,
            ax=ax,
            row_dividers=True,
            footer_divider=True,
            textprops={"fontsize": 10, "ha": "center"},
            row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
            col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
            column_border_kw={"linewidth": 1, "linestyle": "-"},
            index_col="Method",
        ).autoset_fontcolors(
            colnames=[c for c in plot_df.columns if c != "Method"]
        )

    if show:
        plt.show()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for ext in ("svg", "png"):
            fig.savefig(
                os.path.join(save_dir, f"{save_fig_name}.{ext}"),
                facecolor=ax.get_facecolor(),
                dpi=300,
                bbox_inches="tight",
            )
        print(f"Saved to {save_dir}")

    return tab



if __name__ == "__main__":
    
    index_order = ["PCA","Harmony","scGPT","scFoundation","Geneformer","Nicheformer",'OminiCell']
    df = pd.read_csv("/home/cavin/jt/benchmark/experiments/results/cluster_metrics/breast_cancer/human_breast_cancer_metrics_mean.csv", index_col=0,header=0)
    df = df.loc[index_order]

    plot_clustering_eval(df, mean_value_col="mean value", show=True, save_dir="/home/cavin/jt/benchmark/experiments/results/figs/hbc")