import os

from multi_agent_bandits.experiments.thesis_social_trading.common import (
    DEFAULT_RESULTS_ROOT,
    SUMMARY_METRICS,
    aggregate_rows,
    expand_grid,
    run_scenarios,
    write_csv,
)
from multi_agent_bandits.experiments.thesis_social_trading.plotting import plot_sq1_summary


def main(steps=400, seeds=None, save_dir=None):
    seeds = list(seeds or [1, 7, 21, 42, 84])
    output_root = save_dir or DEFAULT_RESULTS_ROOT

    scenario_rows = []
    scenario_rows.append(
        {
            "scenario": "sq1_global_baseline",
            "communication_structure": "global",
            "network_topology": "fully_connected",
            "communication_noise": 0.0,
            "use_reputation": False,
            "malicious_agent_ratio": 0.0,
            "lying_probability": 0.0,
            "lie_magnitude": 0.0,
        }
    )

    for row in expand_grid(
        {
            "network_topology": ["ring_lattice", "random_graph", "fully_connected"],
            "communication_noise": [0.0, 0.15, 0.3, 0.6],
        }
    ):
        scenario_rows.append(
            {
                "scenario": "sq1_local_returns",
                "communication_structure": "local",
                "use_reputation": False,
                "malicious_agent_ratio": 0.0,
                "lying_probability": 0.0,
                "lie_magnitude": 0.0,
                **row,
            }
        )

    for noise_level in [0.15, 0.3, 0.6]:
        scenario_rows.append(
            {
                "scenario": "sq1_global_returns",
                "communication_structure": "global",
                "network_topology": "fully_connected",
                "communication_noise": noise_level,
                "use_reputation": False,
                "malicious_agent_ratio": 0.0,
                "lying_probability": 0.0,
                "lie_magnitude": 0.0,
            }
        )

    output_dir, summary_rows, _ = run_scenarios(
        "sq1_communication_returns",
        scenario_rows,
        steps,
        seeds,
        output_root,
    )

    group_keys = [
        "scenario",
        "communication_structure",
        "network_topology",
        "communication_noise",
    ]
    summary_aggregate = aggregate_rows(summary_rows, group_keys, SUMMARY_METRICS)
    write_csv(
        os.path.join(output_dir, "sq1_summary_by_condition.csv"),
        summary_aggregate,
    )
    plot_sq1_summary(output_dir, summary_aggregate)


if __name__ == "__main__":
    main()
