import csv
import os
import random
from dataclasses import dataclass

from multi_agent_bandits.social_trading.config_social_trading import SocialTradingConfig
from multi_agent_bandits.social_trading.plots_social_trading import plot_single_run
from multi_agent_bandits.social_trading.reputation import ReputationModel
from multi_agent_bandits.social_trading.social_agent import SocialTradingAgent
from multi_agent_bandits.social_trading.social_metrics import compute_summary_metrics
from multi_agent_bandits.social_trading.social_network import SocialNetwork


@dataclass
class SocialSimulationResult:
    config: SocialTradingConfig
    choices_log: list
    rewards_log: list
    reputation_log: list
    pairwise_trust_log: list
    lying_log: list
    cumulative_rewards: list
    summary_metrics: dict
    timestep_metrics: list
    network_edges: list


class SocialTradingSimulation:
    """
    Social-trading simulation that extends the base bandit framework without
    changing the existing core environment.
    """

    def __init__(self, config, agents=None):
        self.config = config
        self.config.validate()

        if self.config.seed is not None:
            random.seed(self.config.seed)

        self.agents = agents or self._build_default_agents()
        self.reputation_model = ReputationModel(
            self.config.n_agents,
            enabled=self.config.use_reputation,
            strength=self.config.reputation_strength,
        )
        self.network = self._build_network()
        self.previous_messages = []
        self.choices_log = []
        self.rewards_log = []
        self.reputation_log = []
        self.pairwise_trust_log = []
        self.lying_log = []
        self.cumulative_rewards = [0.0] * self.config.n_agents

        if self.config.save_dir:
            os.makedirs(self.config.save_dir, exist_ok=True)

    def _build_default_agents(self):
        agents = []
        communication_is_possible = self.config.communication_structure != "none"
        malicious_ratio = (
            self.config.malicious_agent_ratio if communication_is_possible else 0.0
        )
        malicious_count = int(round(self.config.n_agents * malicious_ratio))
        malicious_agents = set(random.sample(range(self.config.n_agents), malicious_count))

        for agent_idx in range(self.config.n_agents):
            agents.append(
                SocialTradingAgent(
                    self.config.n_arms,
                    ucb_exploration=self.config.ucb_exploration,
                    social_influence_strength=self.config.social_influence_strength,
                    crowding_penalty=self.config.crowding_penalty,
                    lying_probability=(
                        self.config.lying_probability
                        if agent_idx in malicious_agents
                        else 0.0
                    ),
                    lie_magnitude=(
                        self.config.lie_magnitude if agent_idx in malicious_agents else 0.0
                    ),
                    name=f"SocialTradingAgent_{agent_idx}",
                )
            )
        return agents

    def _build_network(self):
        if self.config.communication_structure != "local":
            return None

        return SocialNetwork.from_topology(
            self.config.network_topology,
            self.config.n_agents,
            edge_prob=self.config.random_edge_prob,
            neighbors_per_side=self.config.ring_neighbors,
            rng=random,
        )

    def _messages_for_agent(self, agent_idx, current_messages=None):
        current_messages = current_messages or []

        if self.config.communication_structure == "none":
            return []

        visible_messages = self.previous_messages + current_messages

        if self.config.communication_structure == "global":
            return [
                message
                for message in visible_messages
                if message["sender_id"] != agent_idx
            ]

        local_neighbors = set(self.network.neighbors(agent_idx))
        return [
            message
            for message in visible_messages
            if message["sender_id"] in local_neighbors
        ]

    def _aggregate_social_signal(self, receiver_idx, messages):
        signal = [0.0] * self.config.n_arms
        weights = [0.0] * self.config.n_arms
        observed_counts = [0] * self.config.n_arms

        for message in messages:
            arm_idx = message["arm"]
            weight = self.reputation_model.influence_weight(
                receiver_idx,
                message["sender_id"],
            )
            signal[arm_idx] += weight * message["reported_value"]
            weights[arm_idx] += weight
            if message.get("message_type") == "preview":
                observed_counts[arm_idx] += 1

        for arm_idx in range(self.config.n_arms):
            if weights[arm_idx] > 0:
                signal[arm_idx] /= weights[arm_idx]

        return signal, observed_counts

    def _sample_reward(self, arm_idx):
        return self.config.arms[arm_idx].sample()

    def _resolve_rewards(self, choices):
        collisions = {}
        for agent_idx, arm_idx in enumerate(choices):
            collisions.setdefault(arm_idx, []).append(agent_idx)

        rewards = [0.0] * self.config.n_agents
        for arm_idx, agent_ids in collisions.items():
            raw_reward = self._sample_reward(arm_idx)
            if len(agent_ids) == 1:
                rewards[agent_ids[0]] = raw_reward
            else:
                shares = self.config.collision_policy(raw_reward, len(agent_ids))
                for local_idx, agent_id in enumerate(agent_ids):
                    rewards[agent_id] = shares[local_idx]
        return rewards

    def _build_preview_message(
        self,
        agent_idx,
        agent,
        choice,
        social_signal,
        observed_counts,
    ):
        signal = agent.generate_signal(
            choice,
            social_signal=social_signal,
            observed_counts=observed_counts,
        )
        return {
            "sender_id": agent_idx,
            "arm": choice,
            "message_type": "preview",
            "reported_value": signal["reported_value"],
            "truthful_value": signal["truthful_value"],
            "lied": signal["lied"],
            "distortion": signal["distortion"],
        }

    def _build_outcome_messages(self, decision_messages, choices, rewards):
        messages = []
        for agent_idx, (arm_idx, reward) in enumerate(zip(choices, rewards)):
            decision_message = decision_messages[agent_idx]
            reported_value = reward + decision_message["distortion"]
            if self.config.communication_noise > 0:
                reported_value += random.gauss(0.0, self.config.communication_noise)

            messages.append(
                {
                    "sender_id": agent_idx,
                    "arm": arm_idx,
                    "message_type": "outcome",
                    "reported_value": reported_value,
                    "truthful_value": reward,
                    "lied": decision_message["lied"],
                }
            )
        return messages

    def run(self, save_outputs=True, save_plots=False):
        for timestep in range(1, self.config.timesteps + 1):
            choices = []
            decision_messages = []
            observed_messages = []
            for agent_idx, agent in enumerate(self.agents):
                messages = self._messages_for_agent(
                    agent_idx,
                    current_messages=decision_messages,
                )
                observed_messages.append([dict(message) for message in messages])
                social_signal, observed_counts = self._aggregate_social_signal(
                    agent_idx,
                    messages,
                )
                choice = agent.choose_arm(
                    social_signal=social_signal,
                    observed_counts=observed_counts,
                )
                choices.append(choice)
                decision_messages.append(
                    self._build_preview_message(
                        agent_idx,
                        agent,
                        choice,
                        social_signal=social_signal,
                        observed_counts=observed_counts,
                    )
                )

            rewards = self._resolve_rewards(choices)

            for agent_idx, (agent, choice, reward) in enumerate(
                zip(self.agents, choices, rewards)
            ):
                agent.update(reward, chosen_arm=choice)
                self.cumulative_rewards[agent_idx] += reward

            reputations = self.reputation_model.update(
                observed_messages=observed_messages,
                rewards=rewards,
                choices=choices,
            )
            self.previous_messages = self._build_outcome_messages(
                decision_messages,
                choices,
                rewards,
            )

            self.choices_log.append(list(choices))
            self.rewards_log.append(list(rewards))
            self.reputation_log.append(list(reputations))
            self.pairwise_trust_log.append(self.reputation_model.current_pairwise_trust())
            self.lying_log.append([int(message["lied"]) for message in decision_messages])

        arm_means = [arm.mean for arm in self.config.arms]
        summary_metrics, timestep_metrics = compute_summary_metrics(
            self.choices_log,
            self.rewards_log,
            self.cumulative_rewards,
            arm_means,
            self.config.collision_policy,
        )
        summary_metrics["mean_lie_rate"] = (
            sum(sum(row) for row in self.lying_log)
            / (self.config.n_agents * self.config.timesteps)
        )
        summary_metrics["final_mean_reputation"] = (
            sum(self.reputation_log[-1]) / self.config.n_agents
            if self.reputation_log
            else 1.0
        )

        result = SocialSimulationResult(
            config=self.config,
            choices_log=self.choices_log,
            rewards_log=self.rewards_log,
            reputation_log=self.reputation_log,
            pairwise_trust_log=self.pairwise_trust_log,
            lying_log=self.lying_log,
            cumulative_rewards=list(self.cumulative_rewards),
            summary_metrics=summary_metrics,
            timestep_metrics=timestep_metrics,
            network_edges=self.network.edge_list() if self.network is not None else [],
        )

        if save_outputs and self.config.save_dir:
            self.save_outputs(result)

        if save_plots and self.config.save_dir:
            plot_single_run(result, self.config.save_dir)

        return result

    def save_outputs(self, result):
        self._save_matrix(
            os.path.join(self.config.save_dir, "choices.csv"),
            result.choices_log,
        )
        self._save_matrix(
            os.path.join(self.config.save_dir, "rewards.csv"),
            result.rewards_log,
        )
        self._save_matrix(
            os.path.join(self.config.save_dir, "reputations.csv"),
            result.reputation_log,
        )
        self._save_pairwise_trust(
            os.path.join(self.config.save_dir, "pairwise_trust.csv"),
            result.pairwise_trust_log,
        )
        self._save_matrix(
            os.path.join(self.config.save_dir, "lies.csv"),
            result.lying_log,
        )
        self._save_summary_metrics(
            os.path.join(self.config.save_dir, "summary_metrics.csv"),
            result.summary_metrics,
        )
        self._save_timestep_metrics(
            os.path.join(self.config.save_dir, "timestep_metrics.csv"),
            result.timestep_metrics,
        )
        self._save_network(
            os.path.join(self.config.save_dir, "network.csv"),
            result.network_edges,
        )

    def _save_matrix(self, path, rows):
        with open(path, "w", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow([f"agent_{idx}" for idx in range(self.config.n_agents)])
            writer.writerows(rows)

    def _save_summary_metrics(self, path, summary_metrics):
        with open(path, "w", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["metric", "value"])
            for key, value in summary_metrics.items():
                writer.writerow([key, value])

    def _save_pairwise_trust(self, path, trust_rows):
        if not trust_rows:
            return

        headers = []
        for receiver_idx in range(self.config.n_agents):
            for sender_idx in range(self.config.n_agents):
                headers.append(f"trust_{receiver_idx}_to_{sender_idx}")

        flat_rows = []
        for trust_matrix in trust_rows:
            flat_row = []
            for receiver_idx in range(self.config.n_agents):
                for sender_idx in range(self.config.n_agents):
                    flat_row.append(trust_matrix[receiver_idx][sender_idx])
            flat_rows.append(flat_row)

        with open(path, "w", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(headers)
            writer.writerows(flat_rows)

    def _save_timestep_metrics(self, path, timestep_metrics):
        if not timestep_metrics:
            return

        with open(path, "w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(timestep_metrics[0].keys()))
            writer.writeheader()
            writer.writerows(timestep_metrics)

    def _save_network(self, path, network_edges):
        with open(path, "w", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["source", "target"])
            writer.writerows(network_edges)
