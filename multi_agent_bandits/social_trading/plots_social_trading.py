import os

import matplotlib.pyplot as plt


def plot_single_run(result, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    timestep_rows = result.timestep_metrics
    x_values = [row["timestep"] for row in timestep_rows]

    plt.figure(figsize=(10, 4))
    plt.plot(x_values, [row["mean_reward"] for row in timestep_rows], label="Mean reward")
    plt.plot(
        x_values,
        [row["group_reward"] for row in timestep_rows],
        label="Group reward",
        alpha=0.7,
    )
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title("Reward dynamics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_dynamics.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(
        x_values,
        [row["most_popular_arm_share"] for row in timestep_rows],
        label="Most popular arm share",
    )
    plt.plot(
        x_values,
        [row["choice_entropy"] for row in timestep_rows],
        label="Choice entropy",
    )
    plt.plot(
        x_values,
        [row["distinct_arms_chosen"] for row in timestep_rows],
        label="Distinct arms chosen",
    )
    plt.xlabel("Timestep")
    plt.ylabel("Coordination metric")
    plt.title("Coordination over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "coordination_metrics.png"))
    plt.close()

    final_returns = result.cumulative_rewards
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(final_returns)), final_returns)
    plt.xlabel("Agent")
    plt.ylabel("Cumulative reward")
    plt.title("Final cumulative returns by agent")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final_returns.png"))
    plt.close()


def plot_sweep_summary(summary_rows, output_dir):
    if not summary_rows:
        return

    os.makedirs(output_dir, exist_ok=True)

    communication_groups = {}
    for row in summary_rows:
        communication_groups.setdefault(row["communication_structure"], [])
        communication_groups[row["communication_structure"]].append(row)

    plt.figure(figsize=(8, 5))
    for label, rows in communication_groups.items():
        x_values = [row["gini_coefficient"] for row in rows]
        y_values = [row["average_cumulative_return"] for row in rows]
        plt.scatter(x_values, y_values, label=label, alpha=0.7)
    plt.xlabel("Gini coefficient")
    plt.ylabel("Average cumulative return")
    plt.title("Performance vs inequality")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sweep_tradeoff.png"))
    plt.close()
