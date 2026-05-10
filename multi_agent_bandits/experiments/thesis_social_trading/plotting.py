import math
import os
import re

import matplotlib.pyplot as plt


COLORS = {
    "navy": "#0f2d52",
    "blue": "#2f6fed",
    "teal": "#1f8a8a",
    "gold": "#c58b00",
    "rose": "#b54a62",
    "green": "#3a7d44",
    "slate": "#5a6872",
    "violet": "#6f4dbf",
    "orange": "#d0692c",
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

    _plot_sq1_metric_vs_noise(
        output_dir,
        summary_rows,
        "average_cumulative_return_mean",
        "average_cumulative_return_std",
        "SQ1: Return Sensitivity to Communication Noise",
        "Average cumulative return",
        "sq1_returns_vs_noise.png",
    )
    _plot_sq1_local_heatmaps(output_dir, summary_rows)
    _plot_sq1_metric_vs_noise(
        output_dir,
        summary_rows,
        "expected_regret_mean",
        "expected_regret_std",
        "SQ1: Expected Regret Across Noise Levels",
        "Expected regret",
        "sq1_expected_regret_vs_noise.png",
    )
    _plot_sq1_metric_vs_noise(
        output_dir,
        summary_rows,
        "cumulative_return_std_mean",
        "cumulative_return_std_std",
        "SQ1: Return Spread Across Agents",
        "Return standard deviation",
        "sq1_return_spread_vs_noise.png",
    )
    _plot_sq1_metric_vs_noise(
        output_dir,
        summary_rows,
        "gini_coefficient_mean",
        "gini_coefficient_std",
        "SQ1: Inequality Across Noise Levels",
        "Gini coefficient",
        "sq1_gini_vs_noise.png",
    )
    _plot_sq1_metric_vs_noise(
        output_dir,
        summary_rows,
        "mean_choice_entropy_mean",
        "mean_choice_entropy_std",
        "SQ1: Choice Diversity Across Noise Levels",
        "Mean choice entropy",
        "sq1_entropy_vs_noise.png",
    )
    _plot_sq1_metric_vs_noise(
        output_dir,
        summary_rows,
        "mean_most_popular_arm_share_mean",
        "mean_most_popular_arm_share_std",
        "SQ1: Herding Across Noise Levels",
        "Mean most popular arm share",
        "sq1_herding_vs_noise.png",
    )
    _plot_sq1_metric_vs_noise(
        output_dir,
        summary_rows,
        "mean_distinct_arms_chosen_mean",
        "mean_distinct_arms_chosen_std",
        "SQ1: Distinct Arms Used Across Noise Levels",
        "Mean distinct arms chosen",
        "sq1_distinct_arms_vs_noise.png",
    )
    _plot_scatter(
        output_dir,
        summary_rows,
        "expected_regret_mean",
        "average_cumulative_return_mean",
        "SQ1: Return-Regret Tradeoff",
        "Expected regret",
        "Average cumulative return",
        "sq1_return_regret_tradeoff.png",
        _sq1_label,
        _sq1_color,
    )
    _plot_scatter(
        output_dir,
        summary_rows,
        "gini_coefficient_mean",
        "average_cumulative_return_mean",
        "SQ1: Performance-Inequality Tradeoff",
        "Gini coefficient",
        "Average cumulative return",
        "sq1_performance_inequality_tradeoff.png",
        _sq1_label,
        _sq1_color,
    )
    _plot_ranked_bars(
        output_dir,
        summary_rows,
        "average_cumulative_return_mean",
        "average_cumulative_return_std",
        "SQ1: Ranked Average Cumulative Returns",
        "Average cumulative return",
        "sq1_ranked_returns.png",
        reverse=True,
        label_fn=_sq1_label,
        color_fn=_sq1_color,
    )
    _plot_ranked_bars(
        output_dir,
        summary_rows,
        "expected_regret_mean",
        "expected_regret_std",
        "SQ1: Ranked Expected Regret",
        "Expected regret",
        "sq1_ranked_regret.png",
        reverse=False,
        label_fn=_sq1_label,
        color_fn=_sq1_color,
    )
    _plot_sq1_noise_penalty(output_dir, summary_rows)


def plot_sq2_summary(output_dir, summary_rows, timestep_rows):
    apply_thesis_style()
    os.makedirs(output_dir, exist_ok=True)

    _plot_sq2_coordination_overview(output_dir, timestep_rows)
    _plot_sq2_herding_diversification_map(output_dir, summary_rows)
    _plot_sq2_single_metric_trajectories(
        output_dir,
        timestep_rows,
        "group_reward_mean",
        "SQ2: Group Reward Trajectories",
        "Group reward",
        "sq2_group_reward_trajectories.png",
    )
    _plot_sq2_single_metric_trajectories(
        output_dir,
        timestep_rows,
        "choice_entropy_mean",
        "SQ2: Choice Entropy Trajectories",
        "Choice entropy",
        "sq2_choice_entropy_trajectories.png",
    )
    _plot_sq2_single_metric_trajectories(
        output_dir,
        timestep_rows,
        "most_popular_arm_share_mean",
        "SQ2: Herding Trajectories",
        "Most popular arm share",
        "sq2_herding_trajectories.png",
    )
    _plot_sq2_single_metric_trajectories(
        output_dir,
        timestep_rows,
        "distinct_arms_chosen_mean",
        "SQ2: Distinct Arm Trajectories",
        "Distinct arms chosen",
        "sq2_distinct_arms_trajectories.png",
    )
    _plot_sq2_clean_malicious_panels(output_dir, timestep_rows, "group_reward_mean")
    _plot_sq2_clean_malicious_panels(output_dir, timestep_rows, "choice_entropy_mean")
    _plot_ranked_bars(
        output_dir,
        summary_rows,
        "average_cumulative_return_mean",
        "average_cumulative_return_std",
        "SQ2: Average Cumulative Return by Condition",
        "Average cumulative return",
        "sq2_average_cumulative_return_bars.png",
        reverse=True,
        label_fn=_sq2_label,
        color_fn=_sq2_color,
    )
    _plot_ranked_bars(
        output_dir,
        summary_rows,
        "expected_regret_mean",
        "expected_regret_std",
        "SQ2: Expected Regret by Condition",
        "Expected regret",
        "sq2_expected_regret_bars.png",
        reverse=False,
        label_fn=_sq2_label,
        color_fn=_sq2_color,
    )
    _plot_ranked_bars(
        output_dir,
        summary_rows,
        "gini_coefficient_mean",
        "gini_coefficient_std",
        "SQ2: Inequality by Condition",
        "Gini coefficient",
        "sq2_gini_bars.png",
        reverse=False,
        label_fn=_sq2_label,
        color_fn=_sq2_color,
    )
    _plot_ranked_bars(
        output_dir,
        summary_rows,
        "final_choice_entropy_mean",
        "final_choice_entropy_std",
        "SQ2: Final Choice Entropy by Condition",
        "Final choice entropy",
        "sq2_final_entropy_bars.png",
        reverse=True,
        label_fn=_sq2_label,
        color_fn=_sq2_color,
    )
    _plot_ranked_bars(
        output_dir,
        summary_rows,
        "final_most_popular_arm_share_mean",
        "final_most_popular_arm_share_std",
        "SQ2: Final Herding by Condition",
        "Final most popular arm share",
        "sq2_final_herding_bars.png",
        reverse=False,
        label_fn=_sq2_label,
        color_fn=_sq2_color,
    )
    _plot_ranked_bars(
        output_dir,
        summary_rows,
        "mean_lie_rate_mean",
        "mean_lie_rate_std",
        "SQ2: Mean Lie Rate by Condition",
        "Mean lie rate",
        "sq2_mean_lie_rate_bars.png",
        reverse=True,
        label_fn=_sq2_label,
        color_fn=_sq2_color,
    )
    _plot_ranked_bars(
        output_dir,
        summary_rows,
        "final_mean_reputation_mean",
        "final_mean_reputation_std",
        "SQ2: Final Mean Reputation by Condition",
        "Final mean reputation",
        "sq2_final_reputation_bars.png",
        reverse=True,
        label_fn=_sq2_label,
        color_fn=_sq2_color,
    )
    _plot_sq2_malicious_penalty(output_dir, summary_rows)
    _plot_sq2_reputation_gain(output_dir, summary_rows)


def _plot_sq1_metric_vs_noise(
    output_dir,
    summary_rows,
    metric_key,
    std_key,
    title,
    ylabel,
    filename,
):
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for label, color, rows in _sq1_series(summary_rows):
        rows = [row for row in rows if _is_number(row.get(metric_key))]
        if not rows:
            continue
        x_values = [row["communication_noise"] for row in rows]
        y_values = [row[metric_key] for row in rows]
        y_std = [row.get(std_key, 0.0) or 0.0 for row in rows]
        ax.plot(x_values, y_values, color=color, linewidth=2.3, marker="o", label=label)
        ax.fill_between(
            x_values,
            [value - std for value, std in zip(y_values, y_std)],
            [value + std for value, std in zip(y_values, y_std)],
            color=color,
            alpha=0.10,
        )

    ax.set_title(title)
    ax.set_xlabel("Communication noise")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=220)
    plt.close(fig)


def _plot_sq1_local_heatmaps(output_dir, summary_rows):
    local_rows = [
        row for row in summary_rows if row["communication_structure"] == "local"
    ]
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
                if _is_number(value):
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


def _plot_sq1_noise_penalty(output_dir, summary_rows):
    penalty_rows = []
    for label, color, rows in _sq1_series(summary_rows):
        by_noise = {row["communication_noise"]: row for row in rows}
        if 0.0 not in by_noise or 0.6 not in by_noise:
            continue
        clean_return = by_noise[0.0].get("average_cumulative_return_mean")
        noisy_return = by_noise[0.6].get("average_cumulative_return_mean")
        if not _is_number(clean_return) or not _is_number(noisy_return):
            continue
        penalty_rows.append(
            {
                "label": label,
                "penalty": clean_return - noisy_return,
                "color": color,
            }
        )

    if not penalty_rows:
        return

    penalty_rows.sort(key=lambda row: row["penalty"], reverse=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [row["label"] for row in penalty_rows]
    values = [row["penalty"] for row in penalty_rows]
    colors = [row["color"] for row in penalty_rows]
    ax.bar(labels, values, color=colors, alpha=0.9)
    ax.set_title("SQ1: Return Loss From High Communication Noise")
    ax.set_ylabel("Return at noise 0.00 minus return at noise 0.60")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sq1_noise_penalty_by_topology.png"), dpi=220)
    plt.close(fig)


def _plot_sq2_coordination_overview(output_dir, timestep_rows):
    metric_layout = [
        ("most_popular_arm_share_mean", "Most popular arm share"),
        ("choice_entropy_mean", "Choice entropy"),
        ("distinct_arms_chosen_mean", "Distinct arms chosen"),
        ("group_reward_mean", "Group reward"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes_flat = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]

    for (metric_key, title), axis in zip(metric_layout, axes_flat):
        _draw_sq2_metric_lines(axis, timestep_rows, metric_key)
        axis.set_title(title)
        axis.set_xlabel("Timestep")

    axes[0][0].set_ylabel("Coordination")
    axes[1][0].set_ylabel("Diversity")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("SQ2: Reputation, Malicious Agents, and Coordination", y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(os.path.join(output_dir, "sq2_coordination_trajectories.png"), dpi=220)
    plt.close(fig)


def _plot_sq2_herding_diversification_map(output_dir, summary_rows):
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for row in sorted(summary_rows, key=_sq2_sort_key):
        x_value = row.get("final_choice_entropy_mean")
        y_value = row.get("final_most_popular_arm_share_mean")
        if not _is_number(x_value) or not _is_number(y_value):
            continue
        ax.scatter(
            x_value,
            y_value,
            s=150,
            color=_sq2_color(row),
            alpha=0.9,
            edgecolors="white",
            linewidths=1.2,
            marker="s" if row["use_reputation"] else "o",
        )
        ax.text(
            x_value + 0.01,
            y_value + 0.005,
            _sq2_label(row),
            fontsize=8.5,
            color=COLORS["navy"],
        )

    ax.set_title("SQ2: Herding vs Diversification End State")
    ax.set_xlabel("Final choice entropy")
    ax.set_ylabel("Final most popular arm share")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sq2_herding_diversification_map.png"), dpi=220)
    plt.close(fig)


def _plot_sq2_single_metric_trajectories(
    output_dir,
    timestep_rows,
    metric_key,
    title,
    ylabel,
    filename,
):
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    _draw_sq2_metric_lines(ax, timestep_rows, metric_key)
    ax.set_title(title)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=220)
    plt.close(fig)


def _plot_sq2_clean_malicious_panels(output_dir, timestep_rows, metric_key):
    title_by_metric = {
        "group_reward_mean": ("Group Reward", "Group reward"),
        "choice_entropy_mean": ("Choice Entropy", "Choice entropy"),
    }
    title_metric, ylabel = title_by_metric[metric_key]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, malicious, panel_title in [
        (axes[0], False, "Clean communication"),
        (axes[1], True, "Malicious agents"),
    ]:
        panel_rows = [row for row in timestep_rows if _is_malicious(row) == malicious]
        _draw_sq2_metric_lines(ax, panel_rows, metric_key)
        ax.set_title(panel_title)
        ax.set_xlabel("Timestep")
        ax.set_ylabel(ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(f"SQ2: {title_metric} Under Clean vs Malicious Conditions", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    filename = f"sq2_{_safe_name(title_metric)}_clean_vs_malicious.png"
    fig.savefig(os.path.join(output_dir, filename), dpi=220)
    plt.close(fig)


def _plot_sq2_malicious_penalty(output_dir, summary_rows):
    rows = []
    for structure in ["global", "local"]:
        for use_reputation in [False, True]:
            clean = _find_sq2_row(summary_rows, structure, use_reputation, False)
            malicious = _find_sq2_row(summary_rows, structure, use_reputation, True)
            if not clean or not malicious:
                continue
            clean_return = clean.get("average_cumulative_return_mean")
            malicious_return = malicious.get("average_cumulative_return_mean")
            if not _is_number(clean_return) or not _is_number(malicious_return):
                continue
            rows.append(
                {
                    "label": f"{structure.title()}, {_rep_text(use_reputation)}",
                    "penalty": clean_return - malicious_return,
                    "color": _sq2_color(malicious),
                }
            )

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(8.8, 5))
    ax.bar(
        [row["label"] for row in rows],
        [row["penalty"] for row in rows],
        color=[row["color"] for row in rows],
        alpha=0.9,
    )
    ax.axhline(0, color=COLORS["slate"], linewidth=1)
    ax.set_title("SQ2: Return Penalty From Malicious Agents")
    ax.set_ylabel("Clean return minus malicious return")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sq2_malicious_return_penalty.png"), dpi=220)
    plt.close(fig)


def _plot_sq2_reputation_gain(output_dir, summary_rows):
    rows = []
    for structure in ["global", "local"]:
        for malicious in [False, True]:
            no_rep = _find_sq2_row(summary_rows, structure, False, malicious)
            rep = _find_sq2_row(summary_rows, structure, True, malicious)
            if not no_rep or not rep:
                continue
            no_rep_return = no_rep.get("average_cumulative_return_mean")
            rep_return = rep.get("average_cumulative_return_mean")
            if not _is_number(no_rep_return) or not _is_number(rep_return):
                continue
            rows.append(
                {
                    "label": f"{structure.title()}, {_deception_text(malicious)}",
                    "gain": rep_return - no_rep_return,
                    "color": _sq2_color(rep),
                }
            )

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(8.8, 5))
    ax.bar(
        [row["label"] for row in rows],
        [row["gain"] for row in rows],
        color=[row["color"] for row in rows],
        alpha=0.9,
    )
    ax.axhline(0, color=COLORS["slate"], linewidth=1)
    ax.set_title("SQ2: Reputation Gain in Return")
    ax.set_ylabel("Reputation return minus no-reputation return")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sq2_reputation_return_gain.png"), dpi=220)
    plt.close(fig)


def _draw_sq2_metric_lines(axis, timestep_rows, metric_key):
    for scenario, rows in _group_by_scenario(timestep_rows):
        rows = [row for row in rows if _is_number(row.get(metric_key))]
        if not rows:
            continue
        rows.sort(key=lambda row: row["timestep"])
        x_values = [row["timestep"] for row in rows]
        y_values = [row[metric_key] for row in rows]
        std_key = metric_key.replace("_mean", "_std")
        y_std = [row.get(std_key, 0.0) or 0.0 for row in rows]
        representative = rows[0]
        axis.plot(
            x_values,
            y_values,
            linewidth=2.0,
            color=_sq2_color(representative),
            linestyle="--" if _is_malicious(representative) else "-",
            label=_sq2_label(representative),
        )
        axis.fill_between(
            x_values,
            [value - std for value, std in zip(y_values, y_std)],
            [value + std for value, std in zip(y_values, y_std)],
            color=_sq2_color(representative),
            alpha=0.08,
        )


def _plot_scatter(
    output_dir,
    rows,
    x_key,
    y_key,
    title,
    xlabel,
    ylabel,
    filename,
    label_fn,
    color_fn,
):
    fig, ax = plt.subplots(figsize=(9, 5.8))
    for row in rows:
        x_value = row.get(x_key)
        y_value = row.get(y_key)
        if not _is_number(x_value) or not _is_number(y_value):
            continue
        ax.scatter(
            x_value,
            y_value,
            s=110,
            color=color_fn(row),
            alpha=0.9,
            edgecolors="white",
            linewidths=1.1,
        )
        ax.text(
            x_value,
            y_value,
            f"  {label_fn(row)}",
            fontsize=8,
            color=COLORS["navy"],
            va="center",
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=220)
    plt.close(fig)


def _plot_ranked_bars(
    output_dir,
    rows,
    metric_key,
    std_key,
    title,
    xlabel,
    filename,
    reverse,
    label_fn,
    color_fn,
):
    metric_rows = [row for row in rows if _is_number(row.get(metric_key))]
    if not metric_rows:
        return
    metric_rows.sort(key=lambda row: row[metric_key], reverse=reverse)

    labels = [label_fn(row) for row in metric_rows]
    values = [row[metric_key] for row in metric_rows]
    errors = [row.get(std_key, 0.0) or 0.0 for row in metric_rows]
    colors = [color_fn(row) for row in metric_rows]

    fig_height = max(5.0, 0.45 * len(metric_rows) + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    ax.barh(labels, values, xerr=errors, color=colors, alpha=0.9)
    if reverse:
        ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    for idx, value in enumerate(values):
        ax.text(value, idx, f" {value:.2f}", va="center", color=COLORS["navy"], fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=220)
    plt.close(fig)


def _sq1_series(summary_rows):
    global_rows = sorted(
        [row for row in summary_rows if row["communication_structure"] == "global"],
        key=lambda row: row["communication_noise"],
    )
    if global_rows:
        yield "Global", COLORS["navy"], global_rows

    topology_styles = {
        "ring_lattice": (COLORS["green"], "Local ring"),
        "random_graph": (COLORS["rose"], "Local random"),
        "fully_connected": (COLORS["blue"], "Local full"),
    }
    for topology, (color, label) in topology_styles.items():
        rows = sorted(
            [
                row
                for row in summary_rows
                if row["communication_structure"] == "local"
                and row["network_topology"] == topology
            ],
            key=lambda row: row["communication_noise"],
        )
        if rows:
            yield label, color, rows


def _sq1_label(row):
    noise = row.get("communication_noise", 0.0)
    if row["communication_structure"] == "global":
        return f"Global n={noise:.2f}"
    topology = row["network_topology"].replace("_", " ")
    return f"{topology.title()} n={noise:.2f}"


def _sq1_color(row):
    if row["communication_structure"] == "global":
        return COLORS["navy"]
    return {
        "ring_lattice": COLORS["green"],
        "random_graph": COLORS["rose"],
        "fully_connected": COLORS["blue"],
    }.get(row["network_topology"], COLORS["slate"])


def _sq2_label(row):
    structure = row["communication_structure"].title()
    rep = _rep_text(row["use_reputation"])
    deception = _deception_text(_is_malicious(row))
    return f"{structure}, {rep}, {deception}"


def _sq2_color(row):
    key = (
        row.get("communication_structure"),
        bool(row.get("use_reputation")),
        _is_malicious(row),
    )
    color_map = {
        ("global", False, False): COLORS["slate"],
        ("global", True, False): COLORS["blue"],
        ("local", False, False): COLORS["gold"],
        ("local", True, False): COLORS["teal"],
        ("global", False, True): COLORS["rose"],
        ("global", True, True): COLORS["green"],
        ("local", False, True): COLORS["orange"],
        ("local", True, True): COLORS["violet"],
    }
    return color_map.get(key, COLORS["slate"])


def _group_by_scenario(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["scenario"], []).append(row)
    for scenario, scenario_rows in sorted(
        grouped.items(),
        key=lambda item: _sq2_sort_key(item[1][0]),
    ):
        yield scenario, scenario_rows


def _find_sq2_row(summary_rows, structure, use_reputation, malicious):
    for row in summary_rows:
        if (
            row.get("communication_structure") == structure
            and bool(row.get("use_reputation")) == use_reputation
            and _is_malicious(row) == malicious
        ):
            return row
    return None


def _sq2_sort_key(row):
    structure_order = {"global": 0, "local": 1}
    return (
        structure_order.get(row.get("communication_structure"), 99),
        1 if _is_malicious(row) else 0,
        1 if row.get("use_reputation") else 0,
        row.get("scenario", ""),
    )


def _rep_text(use_reputation):
    return "reputation" if use_reputation else "no reputation"


def _deception_text(malicious):
    return "malicious" if malicious else "clean"


def _is_malicious(row):
    return (row.get("malicious_agent_ratio") or 0.0) > 0.0


def _is_number(value):
    return isinstance(value, (int, float)) and not math.isnan(value)


def _safe_name(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
