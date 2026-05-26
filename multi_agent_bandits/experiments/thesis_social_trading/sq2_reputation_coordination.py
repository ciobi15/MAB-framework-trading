import os

from multi_agent_bandits.experiments.thesis_social_trading.common import (
    DEFAULT_RESULTS_ROOT,
    SUMMARY_METRICS,
    TIMESTEP_METRICS,
    aggregate_rows,
    aggregate_timestep_rows,
    run_scenarios,
    write_csv,
)
from multi_agent_bandits.experiments.thesis_social_trading.plotting import plot_sq2_summary


def main(steps=4000, seeds=None, save_dir=None):
    # SQ2 asks whether reputation changes coordination dynamics: do agents herd
    # onto the same arm, or diversify across multiple arms over time?
    seeds = list(seeds or [1, 7, 21, 42, 84])
    output_root = save_dir or DEFAULT_RESULTS_ROOT

    # The scenarios compare the same communication structure with reputation
    # turned off versus on, both in clean communication and in a malicious-agent
    # setting. This keeps SQ2 focused on coordination while adding robustness.
    scenario_rows = []
    for communication_structure, network_topology in [
        ("global", "fully_connected"),
        ("local", "ring_lattice"),
    ]:
        for malicious_agent_ratio, lying_probability, lie_magnitude, deception_label in [
            (0.0, 0.0, 0.0, "clean"),
            (0.25, 0.7, 1.0, "malicious"),
        ]:
            for use_reputation, reputation_label in [
                (False, "no_reputation"),
                (True, "reputation"),
            ]:
                scenario = (
                    f"sq2_{communication_structure}_{deception_label}_"
                    f"{reputation_label}"
                )
                row = {
                    "scenario": scenario,
                    "communication_structure": communication_structure,
                    "network_topology": network_topology,
                    "communication_noise": 0.0,
                    "use_reputation": use_reputation,
                    "malicious_agent_ratio": malicious_agent_ratio,
                    "lying_probability": lying_probability,
                    "lie_magnitude": lie_magnitude,
                }
                if use_reputation:
                    row["reputation_strength"] = 1.0
                    row["reputation_learning_rate"] = 0.25
                scenario_rows.append(row)

    output_dir, summary_rows, timestep_rows = run_scenarios(
        "sq2_reputation_coordination",
        scenario_rows,
        steps,
        seeds,
        output_root,
    )

    # Summary metrics describe overall performance and inequality.
    summary_aggregate = aggregate_rows(
        summary_rows,
        [
            "scenario",
            "communication_structure",
            "network_topology",
            "use_reputation",
            "reputation_learning_rate",
            "malicious_agent_ratio",
            "lying_probability",
            "lie_magnitude",
        ],
        SUMMARY_METRICS,
    )

    # Timestep metrics describe arm-choice distribution over time. These are the
    # main SQ2 outputs for herding versus diversification.
    timestep_aggregate = aggregate_timestep_rows(
        timestep_rows,
        [
            "scenario",
            "communication_structure",
            "network_topology",
            "use_reputation",
            "reputation_learning_rate",
            "malicious_agent_ratio",
            "lying_probability",
            "lie_magnitude",
        ],
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
    plot_sq2_summary(output_dir, summary_aggregate, timestep_aggregate)


if __name__ == "__main__":
    main()
