import itertools
import math


def gini_coefficient(values):
    values = list(values)
    n_values = len(values)
    if n_values == 0:
        return 0.0

    min_value = min(values)
    if min_value < 0:
        values = [value - min_value for value in values]

    total = sum(values)
    if total == 0:
        return 0.0

    sorted_values = sorted(values)
    weighted_sum = 0.0
    for idx, value in enumerate(sorted_values, start=1):
        weighted_sum += idx * value

    return (2 * weighted_sum) / (n_values * total) - (n_values + 1) / n_values


def expected_group_reward(choice_profile, arm_means, collision_policy):
    total_reward = 0.0
    collisions = {}

    for arm_idx in choice_profile:
        collisions.setdefault(arm_idx, 0)
        collisions[arm_idx] += 1

    for arm_idx, n_agents in collisions.items():
        mean_reward = arm_means[arm_idx]
        if n_agents == 1:
            total_reward += mean_reward
        else:
            total_reward += sum(collision_policy(mean_reward, n_agents))

    return total_reward


def oracle_expected_reward_per_step(arm_means, n_agents, collision_policy):
    if getattr(collision_policy, "__name__", "") == "linear_share":
        n_distinct_choices = min(n_agents, len(arm_means))
        return sum(sorted(arm_means, reverse=True)[:n_distinct_choices])

    search_space = len(arm_means) ** n_agents
    if search_space > 100000:
        return None

    best_reward = None
    for choice_profile in itertools.product(range(len(arm_means)), repeat=n_agents):
        reward = expected_group_reward(choice_profile, arm_means, collision_policy)
        if best_reward is None or reward > best_reward:
            best_reward = reward
    return best_reward


def compute_timestep_metrics(choices_log, rewards_log, n_arms):
    timestep_rows = []

    for timestep, (choices, rewards) in enumerate(zip(choices_log, rewards_log), start=1):
        counts = [0] * n_arms
        for arm_idx in choices:
            counts[arm_idx] += 1

        n_agents = len(choices)
        shares = [count / n_agents for count in counts]
        entropy = 0.0
        for share in shares:
            if share > 0:
                entropy -= share * math.log(share, 2)

        row = {
            "timestep": timestep,
            "group_reward": sum(rewards),
            "mean_reward": sum(rewards) / n_agents,
            "most_popular_arm_share": max(shares),
            "choice_entropy": entropy,
            "distinct_arms_chosen": sum(1 for count in counts if count > 0),
        }

        for arm_idx, count in enumerate(counts):
            row[f"arm_{arm_idx}_count"] = count
            row[f"arm_{arm_idx}_share"] = shares[arm_idx]

        timestep_rows.append(row)

    return timestep_rows


def compute_summary_metrics(
    choices_log,
    rewards_log,
    cumulative_rewards,
    arm_means,
    collision_policy,
):
    n_agents = len(cumulative_rewards)
    n_steps = len(choices_log)
    timestep_metrics = compute_timestep_metrics(choices_log, rewards_log, len(arm_means))

    mean_reward_over_time = (
        sum(row["mean_reward"] for row in timestep_metrics) / n_steps if n_steps else 0.0
    )
    group_reward_over_time = (
        sum(row["group_reward"] for row in timestep_metrics) / n_steps if n_steps else 0.0
    )
    cumulative_mean = sum(cumulative_rewards) / n_agents if n_agents else 0.0
    variance = 0.0
    if n_agents:
        variance = sum(
            (reward - cumulative_mean) ** 2 for reward in cumulative_rewards
        ) / n_agents

    summary = {
        "average_cumulative_return": cumulative_mean,
        "average_reward_over_time": mean_reward_over_time,
        "final_group_performance": sum(cumulative_rewards),
        "average_group_reward_per_timestep": group_reward_over_time,
        "cumulative_return_std": math.sqrt(variance),
        "cumulative_return_variance": variance,
        "gini_coefficient": gini_coefficient(cumulative_rewards),
        "final_most_popular_arm_share": (
            timestep_metrics[-1]["most_popular_arm_share"] if timestep_metrics else 0.0
        ),
        "final_choice_entropy": (
            timestep_metrics[-1]["choice_entropy"] if timestep_metrics else 0.0
        ),
        "final_distinct_arms_chosen": (
            timestep_metrics[-1]["distinct_arms_chosen"] if timestep_metrics else 0.0
        ),
        "mean_most_popular_arm_share": (
            sum(row["most_popular_arm_share"] for row in timestep_metrics) / n_steps
            if n_steps
            else 0.0
        ),
        "mean_choice_entropy": (
            sum(row["choice_entropy"] for row in timestep_metrics) / n_steps
            if n_steps
            else 0.0
        ),
        "mean_distinct_arms_chosen": (
            sum(row["distinct_arms_chosen"] for row in timestep_metrics) / n_steps
            if n_steps
            else 0.0
        ),
    }

    oracle_reward = oracle_expected_reward_per_step(
        arm_means,
        n_agents,
        collision_policy,
    )
    if oracle_reward is not None:
        summary["oracle_expected_group_reward_per_step"] = oracle_reward
        summary["expected_regret"] = oracle_reward * n_steps - sum(cumulative_rewards)
    else:
        summary["oracle_expected_group_reward_per_step"] = None
        summary["expected_regret"] = None

    return summary, timestep_metrics
