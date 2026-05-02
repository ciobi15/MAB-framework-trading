import math
import os

import matplotlib.pyplot as plt


COLORS = {
    "navy": "#0f2d52",
    "blue": "#2f6fed",
    "teal": "#1f8a8a",
    "gold": "#c58b00",
    "rose": "#b54a62",
    "green": "#3a7d44",
    "slate": "#5a6872",
    "light": "#edf2f7",
    "grid": "#d9e2ec",
}


def apply_thesis_style():
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["slate"],
            "axes.labelcolor": COLORS["navy"],
            "axes.titlecolor": COLORS["navy"],
            "xtick.color": COLORS["slate"],
            "ytick.color": COLORS["slate"],
            "grid.color": COLORS["grid"],
            "grid.linestyle": "--",
            "grid.alpha": 0.5,
            "axes.grid": True,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )


def plot_sq1_summary(output_dir, summary_rows):
    apply_thesis_style()
    os.makedirs(output_dir, exist_ok=True)

    global_rows = sorted(
        [row for row in summary_rows if row["communication_structure"] == "global"],
        key=lambda row: row["communication_noise"],
    )
    local_rows = [
        row for row in summary_rows if row["communication_structure"] == "local"
    ]

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    if global_rows:
        x_values = [row["communication_noise"] for row in global_rows]
        y_values = [row["average_cumulative_return_mean"] for row in global_rows]
        y_std = [row["average_cumulative_return_std"] for row in global_rows]
        ax.plot(
            x_values,
            y_values,
            color=COLORS["navy"],
            linewidth=2.5,
            marker="o",
            label="Global communication",
        )
        ax.fill_between(
            x_values,
            [value - std for value, std in zip(y_values, y_std)],
            [value + std for value, std in zip(y_values, y_std)],
            color=COLORS["navy"],
            alpha=0.12,
        )

    topology_styles = {
        "ring_lattice": (COLORS["green"], "Ring lattice"),
        "random_graph": (COLORS["rose"], "Random graph"),
        "fully_connected": (COLORS["blue"], "Local fully connected"),
    }
    for topology, (color, label) in topology_styles.items():
        rows = sorted(
            [row for row in local_rows if row["network_topology"] == topology],
            key=lambda row: row["communication_noise"],
        )
        if not rows:
            continue
        x_values = [row["communication_noise"] for row in rows]
        y_values = [row["average_cumulative_return_mean"] for row in rows]
        y_std = [row["average_cumulative_return_std"] for row in rows]
        ax.plot(
            x_values,
            y_values,
            color=color,
            linewidth=2.0,
            marker="o",
            label=label,
        )
        ax.fill_between(
            x_values,
            [value - std for value, std in zip(y_values, y_std)],
            [value + std for value, std in zip(y_values, y_std)],
            color=color,
            alpha=0.10,
        )

    ax.set_title("SQ1: Return Sensitivity to Communication Noise")
    ax.set_xlabel("Communication noise")
    ax.set_ylabel("Average cumulative return")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sq1_returns_vs_noise.png"), dpi=220)
    plt.close(fig)

    local_topologies = ["ring_lattice", "random_graph", "fully_connected"]
    noise_levels = sorted({row["communication_noise"] for row in local_rows})
    metrics = [
        ("average_cumulative_return_mean", "Average cumulative return"),
        ("cumulative_return_std_mean", "Return spread across agents"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (metric_key, title) in zip(axes, metrics):
        matrix = []
        for topology in local_topologies:
            row_values = []
            for noise_level in noise_levels:
                matching = [
                    row
                    for row in local_rows
                    if row["network_topology"] == topology
                    and row["communication_noise"] == noise_level
                ]
                row_values.append(matching[0][metric_key] if matching else math.nan)
            matrix.append(row_values)

        image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(len(noise_levels)))
        ax.set_xticklabels([f"{noise_level:.2f}" for noise_level in noise_levels])
        ax.set_yticks(range(len(local_topologies)))
        ax.set_yticklabels(["Ring", "Random", "Full"])
        ax.set_xlabel("Noise")
        for row_idx, row_values in enumerate(matrix):
            for col_idx, value in enumerate(row_values):
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=9,
                )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("SQ1: Local Communication Landscape", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sq1_local_heatmaps.png"), dpi=220)
    plt.close(fig)


def plot_sq2_summary(output_dir, summary_rows, timestep_rows):
    apply_thesis_style()
    os.makedirs(output_dir, exist_ok=True)

    scenario_order = [
        "sq2_global_no_reputation",
        "sq2_global_reputation",
        "sq2_local_no_reputation",
        "sq2_local_reputation",
    ]
    label_map = {
        "sq2_global_no_reputation": "Global, no reputation",
        "sq2_global_reputation": "Global, reputation",
        "sq2_local_no_reputation": "Local, no reputation",
        "sq2_local_reputation": "Local, reputation",
    }
    color_map = {
        "sq2_global_no_reputation": COLORS["slate"],
        "sq2_global_reputation": COLORS["blue"],
        "sq2_local_no_reputation": COLORS["gold"],
        "sq2_local_reputation": COLORS["teal"],
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    metric_layout = [
        ("most_popular_arm_share_mean", "Most popular arm share", axes[0][0]),
        ("choice_entropy_mean", "Choice entropy", axes[0][1]),
        ("distinct_arms_chosen_mean", "Distinct arms chosen", axes[1][0]),
        ("group_reward_mean", "Group reward", axes[1][1]),
    ]

    for metric_key, title, axis in metric_layout:
        for scenario in scenario_order:
            rows = [
                row for row in timestep_rows if row["scenario"] == scenario
            ]
            rows.sort(key=lambda row: row["timestep"])
            if not rows:
                continue
            x_values = [row["timestep"] for row in rows]
            y_values = [row[metric_key] for row in rows]
            std_key = metric_key.replace("_mean", "_std")
            y_std = [row.get(std_key, 0.0) for row in rows]
            axis.plot(
                x_values,
                y_values,
                linewidth=2.0,
                color=color_map[scenario],
                label=label_map[scenario],
            )
            upper = [value + std for value, std in zip(y_values, y_std)]
            lower = [value - std for value, std in zip(y_values, y_std)]
            axis.fill_between(x_values, lower, upper, color=color_map[scenario], alpha=0.10)

        axis.set_title(title)
        axis.set_xlabel("Timestep")

    axes[0][0].set_ylabel("Coordination")
    axes[1][0].set_ylabel("Diversity")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("SQ2: Reputation and Coordination Dynamics", y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(output_dir, "sq2_coordination_trajectories.png"), dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x_values = [row["final_choice_entropy_mean"] for row in summary_rows]
    y_values = [row["final_most_popular_arm_share_mean"] for row in summary_rows]
    for row, x_value, y_value in zip(summary_rows, x_values, y_values):
        scenario = row["scenario"]
        ax.scatter(
            x_value,
            y_value,
            s=130,
            color=color_map.get(scenario, COLORS["blue"]),
            alpha=0.9,
            edgecolors="white",
            linewidths=1.2,
        )
        ax.text(
            x_value + 0.01,
            y_value + 0.005,
            label_map.get(scenario, scenario),
            fontsize=9,
            color=COLORS["navy"],
        )

    ax.set_title("SQ2: Herding vs Diversification End State")
    ax.set_xlabel("Final choice entropy")
    ax.set_ylabel("Final most popular arm share")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sq2_herding_diversification_map.png"), dpi=220)
    plt.close(fig)


def plot_sq3_summary(output_dir, ranked_rows):
    apply_thesis_style()
    os.makedirs(output_dir, exist_ok=True)

    color_map = {
        "none": COLORS["gold"],
        "global": COLORS["blue"],
        "local": COLORS["teal"],
    }

    fig, ax = plt.subplots(figsize=(9.5, 6))
    rows = sorted(
        ranked_rows,
        key=lambda row: row["average_cumulative_return_mean"],
        reverse=True,
    )
    pareto_rows = [row for row in rows if row.get("pareto_efficient")]
    pareto_rows.sort(key=lambda row: row["gini_coefficient_mean"])

    for row in rows:
        color = color_map.get(row["communication_structure"], COLORS["slate"])
        size = 120 + 90 * row["use_reputation"]
        ax.scatter(
            row["gini_coefficient_mean"],
            row["average_cumulative_return_mean"],
            s=size,
            color=color,
            alpha=0.9,
            edgecolors="white",
            linewidths=1.2,
            marker="s" if row["use_reputation"] else "o",
        )
        ax.text(
            row["gini_coefficient_mean"] + 0.002,
            row["average_cumulative_return_mean"] + 1.5,
            row["scenario"].replace("sq3_", ""),
            fontsize=8.5,
            color=COLORS["navy"],
        )

    if pareto_rows:
        ax.plot(
            [row["gini_coefficient_mean"] for row in pareto_rows],
            [row["average_cumulative_return_mean"] for row in pareto_rows],
            color=COLORS["rose"],
            linewidth=2.0,
            linestyle="--",
            label="Pareto frontier",
        )

    ax.set_title("SQ3: Performance-Inequality Frontier")
    ax.set_xlabel("Gini coefficient")
    ax.set_ylabel("Average cumulative return")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sq3_pareto_frontier.png"), dpi=220)
    plt.close(fig)

    rows = sorted(
        ranked_rows,
        key=lambda row: row["tradeoff_score"],
        reverse=True,
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = [row["scenario"].replace("sq3_", "") for row in rows]
    values = [row["tradeoff_score"] for row in rows]
    colors = [color_map.get(row["communication_structure"], COLORS["slate"]) for row in rows]
    ax.barh(labels, values, color=colors, alpha=0.9)
    ax.invert_yaxis()
    ax.set_title("SQ3: Ranked Performance-Inequality Tradeoff")
    ax.set_xlabel("Tradeoff score")
    for idx, value in enumerate(values):
        ax.text(value + 0.01, idx, f"{value:.2f}", va="center", color=COLORS["navy"])
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sq3_tradeoff_ranking.png"), dpi=220)
    plt.close(fig)
