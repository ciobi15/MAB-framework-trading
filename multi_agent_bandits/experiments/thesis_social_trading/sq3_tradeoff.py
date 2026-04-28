import os

from multi_agent_bandits.experiments.thesis_social_trading.common import (
    SUMMARY_METRICS,
    add_tradeoff_score,
    aggregate_rows,
    mark_pareto_efficient,
    run_scenarios,
    sort_by_tradeoff,
    write_csv,
)


def main(steps=400, seeds=None, save_dir=None):
    seeds = list(seeds or [1, 7, 21, 42, 84])
    output_root = save_dir or "results/thesis_social_trading"

    scenario_rows = [
        {
            "scenario": "sq3_pure_ucb",
            "communication_structure": "none",
            "network_topology": "fully_connected",
            "communication_noise": 0.0,
            "use_reputation": False,
            "malicious_agent_ratio": 0.0,
            "lying_probability": 0.0,
            "lie_magnitude": 0.0,
        },
        {
            "scenario": "sq3_global_information",
            "communication_structure": "global",
            "network_topology": "fully_connected",
            "communication_noise": 0.0,
            "use_reputation": False,
            "malicious_agent_ratio": 0.0,
            "lying_probability": 0.0,
            "lie_magnitude": 0.0,
        },
        {
            "scenario": "sq3_global_reputation",
            "communication_structure": "global",
            "network_topology": "fully_connected",
            "communication_noise": 0.0,
            "use_reputation": True,
            "reputation_strength": 1.0,
            "malicious_agent_ratio": 0.0,
            "lying_probability": 0.0,
            "lie_magnitude": 0.0,
        },
        {
            "scenario": "sq3_local_information",
            "communication_structure": "local",
            "network_topology": "ring_lattice",
            "communication_noise": 0.0,
            "use_reputation": False,
            "malicious_agent_ratio": 0.0,
            "lying_probability": 0.0,
            "lie_magnitude": 0.0,
        },
        {
            "scenario": "sq3_local_reputation",
            "communication_structure": "local",
            "network_topology": "ring_lattice",
            "communication_noise": 0.0,
            "use_reputation": True,
            "reputation_strength": 1.0,
            "malicious_agent_ratio": 0.0,
            "lying_probability": 0.0,
            "lie_magnitude": 0.0,
        },
        {
            "scenario": "sq3_local_random_reputation",
            "communication_structure": "local",
            "network_topology": "random_graph",
            "communication_noise": 0.0,
            "use_reputation": True,
            "reputation_strength": 1.0,
            "malicious_agent_ratio": 0.0,
            "lying_probability": 0.0,
            "lie_magnitude": 0.0,
        },
    ]

    output_dir, summary_rows, _ = run_scenarios(
        "sq3_tradeoff",
        scenario_rows,
        steps,
        seeds,
        output_root,
    )

    summary_aggregate = aggregate_rows(
        summary_rows,
        ["scenario", "communication_structure", "network_topology", "use_reputation"],
        SUMMARY_METRICS,
    )
    summary_aggregate = add_tradeoff_score(
        summary_aggregate,
        "average_cumulative_return_mean",
        "gini_coefficient_mean",
    )
    summary_aggregate = mark_pareto_efficient(
        summary_aggregate,
        "average_cumulative_return_mean",
        "gini_coefficient_mean",
    )
    ranked_rows = sort_by_tradeoff(summary_aggregate)

    write_csv(
        os.path.join(output_dir, "sq3_tradeoff_summary.csv"),
        ranked_rows,
    )


if __name__ == "__main__":
    main()
