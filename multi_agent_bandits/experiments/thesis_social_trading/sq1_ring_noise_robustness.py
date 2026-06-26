import csv
import os

from multi_agent_bandits.experiments.thesis_social_trading.common import (
    SUMMARY_METRICS,
    aggregate_rows,
    run_scenarios,
    write_csv,
)


NOISE_LEVELS = [0.0, 0.15, 0.3, 0.6]


def main(steps=4000, seeds=None, save_dir=None):
    # Focused SQ1 robustness check: does moderate communication noise reliably
    # improve ring-lattice performance, or was that a small-seed artifact?
    seeds = list(seeds or range(1, 31))
    output_root = save_dir or "results/thesis_social_trading"

    scenario_rows = [
        {
            "scenario": "sq1_ring_noise_robustness",
            "communication_structure": "local",
            "network_topology": "ring_lattice",
            "communication_noise": noise_level,
            "use_reputation": False,
            "malicious_agent_ratio": 0.0,
            "lying_probability": 0.0,
            "lie_magnitude": 0.0,
        }
        for noise_level in NOISE_LEVELS
    ]

    output_dir, summary_rows, _ = run_scenarios(
        "sq1_ring_noise_robustness",
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
        os.path.join(output_dir, "sq1_ring_noise_summary_by_condition.csv"),
        summary_aggregate,
    )
    write_csv(
        os.path.join(output_dir, "sq1_ring_noise_seed_comparison.csv"),
        seed_comparison_rows(summary_rows),
    )


def seed_comparison_rows(summary_rows):
    rows_by_seed = {}
    for row in summary_rows:
        rows_by_seed.setdefault(row["seed"], {})[row["communication_noise"]] = row

    comparison_rows = []
    for seed, seed_rows in sorted(rows_by_seed.items()):
        baseline = seed_rows.get(0.0)
        moderate = seed_rows.get(0.3)
        high = seed_rows.get(0.6)
        if not baseline or not moderate:
            continue

        baseline_return = baseline["average_cumulative_return"]
        moderate_return = moderate["average_cumulative_return"]
        high_return = high["average_cumulative_return"] if high else None

        comparison_rows.append(
            {
                "seed": seed,
                "return_noise_0_00": baseline_return,
                "return_noise_0_30": moderate_return,
                "delta_0_30_minus_0_00": moderate_return - baseline_return,
                "noise_0_30_beats_0_00": moderate_return > baseline_return,
                "return_noise_0_60": high_return,
                "delta_0_60_minus_0_00": (
                    high_return - baseline_return if high_return is not None else None
                ),
                "noise_0_60_beats_0_00": (
                    high_return > baseline_return if high_return is not None else None
                ),
            }
        )

    return comparison_rows


if __name__ == "__main__":
    main()
