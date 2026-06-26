import os

import matplotlib.pyplot as plt

from multi_agent_bandits.experiments.thesis_social_trading.common import (
    DEFAULT_RESULTS_ROOT,
    SUMMARY_METRICS,
    TIMESTEP_METRICS,
    aggregate_rows,
    aggregate_timestep_rows,
    run_scenarios,
    write_csv,
)
from multi_agent_bandits.experiments.thesis_social_trading.plotting import (
    COLORS,
    apply_thesis_style,
)


LEARNING_RATES = [0.05, 0.1, 0.25, 0.5, 0.75]


def main(steps=4000, seeds=None, save_dir=None):
    # Sensitivity check for SQ2: how quickly should reputation react to new
    # evidence about message accuracy?
    seeds = list(seeds or [1, 7, 21, 42, 84])
    output_root = save_dir or DEFAULT_RESULTS_ROOT

    scenario_rows = []
    for communication_structure, network_topology in [
        ("global", "fully_connected"),
        ("local", "ring_lattice"),
    ]:
        for malicious_agent_ratio, lying_probability, lie_magnitude, deception_label in [
            (0.0, 0.0, 0.0, "clean"),
            (0.25, 0.7, 1.0, "malicious"),
        ]:
            scenario_rows.append(
                {
                    "scenario": (
                        f"sq2_lr_{communication_structure}_{deception_label}_"
                        "no_reputation"
                    ),
                    "communication_structure": communication_structure,
                    "network_topology": network_topology,
                    "communication_noise": 0.0,
                    "use_reputation": False,
                    "malicious_agent_ratio": malicious_agent_ratio,
                    "lying_probability": lying_probability,
                    "lie_magnitude": lie_magnitude,
                }
            )

            for learning_rate in LEARNING_RATES:
                scenario_rows.append(
                    {
                        "scenario": (
                            f"sq2_lr_{communication_structure}_{deception_label}_"
                            f"lr_{learning_rate:g}"
                        ),
                        "communication_structure": communication_structure,
                        "network_topology": network_topology,
                        "communication_noise": 0.0,
                        "use_reputation": True,
                        "reputation_strength": 1.0,
                        "reputation_learning_rate": learning_rate,
                        "malicious_agent_ratio": malicious_agent_ratio,
                        "lying_probability": lying_probability,
                        "lie_magnitude": lie_magnitude,
                    }
                )

    output_dir, summary_rows, timestep_rows = run_scenarios(
        "sq2_reputation_learning_rate_sweep",
        scenario_rows,
        steps,
        seeds,
        output_root,
    )

    group_keys = [
        "scenario",
        "communication_structure",
        "network_topology",
        "use_reputation",
        "reputation_learning_rate",
        "malicious_agent_ratio",
        "lying_probability",
        "lie_magnitude",
    ]
    summary_aggregate = aggregate_rows(summary_rows, group_keys, SUMMARY_METRICS)
    timestep_aggregate = aggregate_timestep_rows(
        timestep_rows,
        group_keys,
        TIMESTEP_METRICS,
    )

    write_csv(
        os.path.join(output_dir, "sq2_reputation_learning_rate_summary.csv"),
        summary_aggregate,
    )
    write_csv(
        os.path.join(output_dir, "sq2_reputation_learning_rate_timestep.csv"),
        timestep_aggregate,
    )
    plot_learning_rate_sweep(output_dir, summary_aggregate)


def plot_learning_rate_sweep(output_dir, summary_rows):
    apply_thesis_style()
    os.makedirs(output_dir, exist_ok=True)
    reputation_rows = [row for row in summary_rows if row["use_reputation"]]

    _plot_learning_rate_metric(
        output_dir,
        reputation_rows,
        "average_cumulative_return_mean",
        "average_cumulative_return_std",
        "SQ2: Return Sensitivity to Reputation Learning Rate",
        "Average cumulative return",
        "sq2_reputation_learning_rate_returns.png",
    )
    _plot_learning_rate_metric(
        output_dir,
        reputation_rows,
        "final_choice_entropy_mean",
        "final_choice_entropy_std",
        "SQ2: Diversification Sensitivity to Reputation Learning Rate",
        "Final choice entropy",
        "sq2_reputation_learning_rate_entropy.png",
    )
    _plot_learning_rate_metric(
        output_dir,
        reputation_rows,
        "final_mean_reputation_mean",
        "final_mean_reputation_std",
        "SQ2: Final Reputation by Learning Rate",
        "Final mean reputation",
        "sq2_reputation_learning_rate_final_reputation.png",
    )


def _plot_learning_rate_metric(
    output_dir,
    rows,
    metric_key,
    std_key,
    title,
    ylabel,
    filename,
):
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for label, color, series_rows in _learning_rate_series(rows):
        x_values = [row["reputation_learning_rate"] for row in series_rows]
        y_values = [row[metric_key] for row in series_rows]
        y_std = [row.get(std_key, 0.0) or 0.0 for row in series_rows]
        ax.plot(x_values, y_values, color=color, linewidth=2.3, marker="o", label=label)
        ax.fill_between(
            x_values,
            [value - std for value, std in zip(y_values, y_std)],
            [value + std for value, std in zip(y_values, y_std)],
            color=color,
            alpha=0.10,
        )

    ax.set_title(title)
    ax.set_xlabel("Reputation learning rate")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(os.path.join(output_dir, filename), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _learning_rate_series(rows):
    series_specs = [
        ("Global clean", "global", False, COLORS["navy"]),
        ("Global malicious", "global", True, COLORS["rose"]),
        ("Local clean", "local", False, COLORS["blue"]),
        ("Local malicious", "local", True, COLORS["green"]),
    ]
    for label, structure, malicious, color in series_specs:
        series_rows = [
            row
            for row in rows
            if row["communication_structure"] == structure
            and _is_malicious(row) == malicious
        ]
        series_rows.sort(key=lambda row: row["reputation_learning_rate"])
        if series_rows:
            yield label, color, series_rows


def _is_malicious(row):
    return (row.get("malicious_agent_ratio") or 0.0) > 0.0


if __name__ == "__main__":
    main()
