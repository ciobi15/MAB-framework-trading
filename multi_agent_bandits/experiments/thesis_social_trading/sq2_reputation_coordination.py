import os

from multi_agent_bandits.experiments.thesis_social_trading.common import (
    SUMMARY_METRICS,
    TIMESTEP_METRICS,
    aggregate_rows,
    aggregate_timestep_rows,
    run_scenarios,
    write_csv,
)


def main(steps=400, seeds=None, save_dir=None):
    seeds = list(seeds or [1, 7, 21, 42, 84])
    output_root = save_dir or "results/thesis_social_trading"

    scenario_rows = [
        {
            "scenario": "sq2_global_no_reputation",
            "communication_structure": "global",
            "network_topology": "fully_connected",
            "communication_noise": 0.0,
            "use_reputation": False,
            "malicious_agent_ratio": 0.0,
            "lying_probability": 0.0,
            "lie_magnitude": 0.0,
        },
        {
            "scenario": "sq2_global_reputation",
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
            "scenario": "sq2_local_no_reputation",
            "communication_structure": "local",
            "network_topology": "ring_lattice",
            "communication_noise": 0.0,
            "use_reputation": False,
            "malicious_agent_ratio": 0.0,
            "lying_probability": 0.0,
            "lie_magnitude": 0.0,
        },
        {
            "scenario": "sq2_local_reputation",
            "communication_structure": "local",
            "network_topology": "ring_lattice",
            "communication_noise": 0.0,
            "use_reputation": True,
            "reputation_strength": 1.0,
            "malicious_agent_ratio": 0.0,
            "lying_probability": 0.0,
            "lie_magnitude": 0.0,
        },
    ]

    output_dir, summary_rows, timestep_rows = run_scenarios(
        "sq2_reputation_coordination",
        scenario_rows,
        steps,
        seeds,
        output_root,
    )

    summary_aggregate = aggregate_rows(
        summary_rows,
        ["scenario", "communication_structure", "use_reputation"],
        SUMMARY_METRICS,
    )
    timestep_aggregate = aggregate_timestep_rows(
        timestep_rows,
        ["scenario", "communication_structure", "use_reputation"],
        TIMESTEP_METRICS,
    )

    write_csv(
        os.path.join(output_dir, "sq2_summary_by_condition.csv"),
        summary_aggregate,
    )
    write_csv(
        os.path.join(output_dir, "sq2_timestep_coordination.csv"),
        timestep_aggregate,
    )


if __name__ == "__main__":
    main()
