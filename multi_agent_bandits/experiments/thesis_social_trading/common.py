import csv
import math
import os
from collections import defaultdict
from itertools import product
from statistics import mean, pstdev

from multi_agent_bandits.core.arm import Arm
from multi_agent_bandits.core.reward_sharing import linear_share
from multi_agent_bandits.social_trading.config_social_trading import SocialTradingConfig
from multi_agent_bandits.social_trading.sweep_runner import SweepRunner

PACKAGE_DIR = os.path.dirname(__file__)
DEFAULT_RESULTS_ROOT = os.path.join(PACKAGE_DIR, "results", "thesis_social_trading")


SUMMARY_METRICS = [
    "average_cumulative_return",
    "average_reward_over_time",
    "average_group_reward_per_timestep",
    "final_group_performance",
    "cumulative_return_std",
    "gini_coefficient",
    "expected_regret",
    "mean_most_popular_arm_share",
    "mean_choice_entropy",
    "mean_distinct_arms_chosen",
    "final_most_popular_arm_share",
    "final_choice_entropy",
    "final_distinct_arms_chosen",
    "mean_lie_rate",
    "final_mean_reputation",
]

TIMESTEP_METRICS = [
    "group_reward",
    "mean_reward",
    "most_popular_arm_share",
    "choice_entropy",
    "distinct_arms_chosen",
]


def build_base_config(steps):
    return SocialTradingConfig(
        n_agents=8,
        arms=[
            Arm(mean=0.8, sd=1.0),
            Arm(mean=1.1, sd=1.0),
            Arm(mean=1.5, sd=1.0),
            Arm(mean=1.9, sd=1.0),
        ],
        timesteps=steps,
        collision_policy=linear_share,
        communication_structure="none",
        network_topology="fully_connected",
        communication_noise=0.0,
        use_reputation=False,
        reputation_strength=1.0,
        social_influence_strength=0.6,
        ucb_exploration=2.0,
        crowding_penalty=0.3,
        malicious_agent_ratio=0.0,
        lying_probability=0.0,
        lie_magnitude=0.0,
    )


def expand_grid(grid):
    names = list(grid.keys())
    for values in product(*(grid[name] for name in names)):
        yield dict(zip(names, values))


def run_scenarios(
    question_name,
    scenario_rows,
    steps,
    seeds,
    output_root,
    save_plots=False,
):
    output_dir = os.path.join(output_root, question_name)
    runner = SweepRunner(
        base_config=build_base_config(steps),
        sweep_parameters={},
        seeds=seeds,
        output_dir=output_dir,
        scenario_rows=scenario_rows,
    )
    summary_rows, timestep_rows = runner.run(save_plots=save_plots)
    return output_dir, summary_rows, timestep_rows


def aggregate_rows(rows, group_keys, metric_keys):
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[group_key] for group_key in group_keys)
        grouped[key].append(row)

    aggregated_rows = []
    for key, group_rows in grouped.items():
        aggregated_row = {
            group_key: key[idx] for idx, group_key in enumerate(group_keys)
        }
        aggregated_row["n_runs"] = len(group_rows)

        for metric_key in metric_keys:
            metric_values = [
                row[metric_key]
                for row in group_rows
                if isinstance(row.get(metric_key), (int, float))
            ]
            if not metric_values:
                aggregated_row[f"{metric_key}_mean"] = None
                aggregated_row[f"{metric_key}_std"] = None
                continue

            aggregated_row[f"{metric_key}_mean"] = mean(metric_values)
            aggregated_row[f"{metric_key}_std"] = (
                pstdev(metric_values) if len(metric_values) > 1 else 0.0
            )

        aggregated_rows.append(aggregated_row)

    aggregated_rows.sort(key=lambda row: tuple(row[group_key] for group_key in group_keys))
    return aggregated_rows


def aggregate_timestep_rows(rows, group_keys, metric_keys):
    timestep_group_keys = list(group_keys) + ["timestep"]
    return aggregate_rows(rows, timestep_group_keys, metric_keys)


def write_csv(path, rows):
    if not rows:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def add_tradeoff_score(rows, performance_key, inequality_key):
    numeric_rows = [
        row
        for row in rows
        if isinstance(row.get(performance_key), (int, float))
        and isinstance(row.get(inequality_key), (int, float))
    ]
    if not numeric_rows:
        return rows

    performance_values = [row[performance_key] for row in numeric_rows]
    inequality_values = [row[inequality_key] for row in numeric_rows]

    min_performance = min(performance_values)
    max_performance = max(performance_values)
    min_inequality = min(inequality_values)
    max_inequality = max(inequality_values)

    for row in rows:
        performance = row.get(performance_key)
        inequality = row.get(inequality_key)
        if not isinstance(performance, (int, float)) or not isinstance(inequality, (int, float)):
            row["tradeoff_score"] = None
            continue

        normalized_performance = _normalize(performance, min_performance, max_performance)
        normalized_inequality = _normalize(inequality, min_inequality, max_inequality)
        row["tradeoff_score"] = normalized_performance - normalized_inequality

    return rows


def mark_pareto_efficient(rows, performance_key, inequality_key):
    for row in rows:
        row["pareto_efficient"] = False

    for candidate_row in rows:
        candidate_performance = candidate_row.get(performance_key)
        candidate_inequality = candidate_row.get(inequality_key)
        if not isinstance(candidate_performance, (int, float)) or not isinstance(
            candidate_inequality,
            (int, float),
        ):
            continue

        dominated = False
        for competitor_row in rows:
            competitor_performance = competitor_row.get(performance_key)
            competitor_inequality = competitor_row.get(inequality_key)
            if not isinstance(competitor_performance, (int, float)) or not isinstance(
                competitor_inequality,
                (int, float),
            ):
                continue

            no_worse = (
                competitor_performance >= candidate_performance
                and competitor_inequality <= candidate_inequality
            )
            strictly_better = (
                competitor_performance > candidate_performance
                or competitor_inequality < candidate_inequality
            )
            if no_worse and strictly_better:
                dominated = True
                break

        candidate_row["pareto_efficient"] = not dominated

    return rows


def sort_by_tradeoff(rows):
    return sorted(
        rows,
        key=lambda row: (
            -row["tradeoff_score"] if isinstance(row.get("tradeoff_score"), (int, float)) else math.inf,
            row.get("gini_coefficient_mean", math.inf),
        ),
    )


def _normalize(value, low, high):
    if abs(high - low) < 1e-12:
        return 0.5
    return (value - low) / (high - low)
