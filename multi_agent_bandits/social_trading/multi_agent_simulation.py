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
    """Container for everything produced by one simulation run."""

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

    The simulation coordinates three things:
    1. agents choosing arms with UCB,
    2. messages shared between agents,
    3. reputation/trust updates based on how useful those messages were.
    """

    def __init__(self, config, agents=None):
        self.config = config
        self.config.validate()

        # A fixed seed makes experiment runs reproducible.
        if self.config.seed is not None:
            random.seed(self.config.seed)

        self.agents = agents or self._build_default_agents()

        # ReputationModel stores the pairwise trust matrix. If reputation is
        # disabled, it still returns neutral trust weights of 1.0.
        self.reputation_model = ReputationModel(
            self.config.n_agents,
            enabled=self.config.use_reputation,
            strength=self.config.reputation_strength,
            learning_rate=self.config.reputation_learning_rate,
        )
        self.network = self._build_network()

        # Outcome messages from the previous timestep are stored here so agents
        # can use last round's results when choosing in the next round.
        self.previous_messages = []

        # Logs are used later for metrics, plots, and CSV outputs.
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

        # Malicious agents only matter when communication exists. If agents
        # cannot send messages, lying would have no effect on other agents.
        malicious_ratio = (
            self.config.malicious_agent_ratio if communication_is_possible else 0.0
        )
        malicious_count = int(round(self.config.n_agents * malicious_ratio))
        malicious_agents = set(random.sample(range(self.config.n_agents), malicious_count))

        for agent_idx in range(self.config.n_agents):
            # Honest agents have lying_probability = 0 and lie_magnitude = 0.
            # Malicious agents use the configured lying parameters.
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
        # A network is only needed for local communication. Global communication
        # directly exposes messages to all agents, and "none" exposes no messages.
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

        # Before choosing, an agent can observe outcome messages from the
        # previous timestep plus preview messages from agents who already chose
        # in the current timestep.
        visible_messages = self.previous_messages + current_messages
        return self._visible_messages_from_pool(agent_idx, visible_messages)

    def _visible_messages_from_pool(self, agent_idx, messages):
        """Filter a message pool according to the communication structure."""
        if self.config.communication_structure == "none":
            return []

        # Global communication: every other agent's messages are visible.
        if self.config.communication_structure == "global":
            return [
                message
                for message in messages
                if message["sender_id"] != agent_idx
            ]

        # Local communication: only messages from network neighbors are visible.
        local_neighbors = set(self.network.neighbors(agent_idx))
        return [
            message
            for message in messages
            if message["sender_id"] in local_neighbors
        ]

    def _aggregate_social_signal(self, receiver_idx, messages):
        """
        Turn visible messages into one social value per arm.

        Messages are weighted by the receiver's trust in the sender. This is
        where reputation affects the later arm choice.
        """
        signal = [0.0] * self.config.n_arms
        weights = [0.0] * self.config.n_arms
        observed_counts = [0] * self.config.n_arms

        for message in messages:
            arm_idx = message["arm"]

            # Pairwise trust becomes the influence weight for this message.
            weight = self.reputation_model.influence_weight(
                receiver_idx,
                message["sender_id"],
            )
            signal[arm_idx] += weight * message["reported_value"]
            weights[arm_idx] += weight

            # Only preview messages count as current crowding, because they say
            # who is choosing an arm right now.
            if message.get("message_type") == "preview":
                observed_counts[arm_idx] += 1

        # Convert weighted sums into weighted averages for each arm.
        for arm_idx in range(self.config.n_arms):
            if weights[arm_idx] > 0:
                signal[arm_idx] /= weights[arm_idx]

        return signal, observed_counts

    def _sample_reward(self, arm_idx):
        return self.config.arms[arm_idx].sample()

    def _resolve_rewards(self, choices):
        """
        Sample rewards after all agents have chosen.

        If multiple agents choose the same arm, the configured collision policy
        decides how the sampled reward is shared between them.
        """
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
        """
        Build a preview message after an agent chooses but before reward is known.

        Preview messages communicate the chosen arm and the sender's current
        estimated value for that arm. Later agents in the same timestep may see
        this message before making their own choice.
        """
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
        """
        Build outcome messages after rewards are known.

        Outcome messages communicate what reward the sender reports receiving.
        These messages update reputation and become previous_messages for the
        next timestep.
        """
        messages = []
        for agent_idx, (arm_idx, reward) in enumerate(zip(choices, rewards)):
            decision_message = decision_messages[agent_idx]

            # The same distortion chosen for the preview message is carried into
            # the outcome report, so a lying agent is consistently misleading in
            # that timestep.
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
            # Per-timestep state. decision_messages are preview messages created
            # during this timestep; observed_messages records what each receiver
            # saw before choosing, so those messages can later update reputation.
            choices = []
            decision_messages = []
            observed_messages = []
            for agent_idx, agent in enumerate(self.agents):
                # Collect the messages visible to this agent before it chooses.
                messages = self._messages_for_agent(
                    agent_idx,
                    current_messages=decision_messages,
                )
                observed_messages.append([dict(message) for message in messages])

                # Convert messages into social_signal and crowding counts, then
                # let the agent combine those with its own UCB estimates.
                social_signal, observed_counts = self._aggregate_social_signal(
                    agent_idx,
                    messages,
                )
                choice = agent.choose_arm(
                    social_signal=social_signal,
                    observed_counts=observed_counts,
                )
                choices.append(choice)

                # Once the agent has chosen, it broadcasts a preview message.
                # Agents later in this same timestep may observe it.
                decision_messages.append(
                    self._build_preview_message(
                        agent_idx,
                        agent,
                        choice,
                        social_signal=social_signal,
                        observed_counts=observed_counts,
                    )
                )

            # Rewards are only sampled after every agent has committed to an arm.
            rewards = self._resolve_rewards(choices)

            # Each agent updates its private bandit estimate using its own reward.
            for agent_idx, (agent, choice, reward) in enumerate(
                zip(self.agents, choices, rewards)
            ):
                agent.update(reward, chosen_arm=choice)
                self.cumulative_rewards[agent_idx] += reward

            # Outcome messages report the rewards after they are known.
            outcome_messages = self._build_outcome_messages(
                decision_messages,
                choices,
                rewards,
            )

            reputation_messages = []
            for agent_idx, messages in enumerate(observed_messages):
                # Messages observed before choosing can be judged once the
                # current rewards are known. This lets an outcome message like
                # "arm 2 paid 10" lose trust if the receiver follows it and
                # receives a much smaller reward.
                observed_before_choice = [
                    dict(message)
                    for message in messages
                ]
                visible_outcomes = [
                    dict(message)
                    for message in self._visible_messages_from_pool(
                        agent_idx,
                        outcome_messages,
                    )
                ]

                # Reputation receives both:
                # - messages the agent used before choosing,
                # - visible outcome messages after rewards were known.
                # This makes trust reflect whether communicated information was
                # useful for the receiver's actual experience.
                reputation_messages.append(observed_before_choice + visible_outcomes)

            # Update the pairwise trust matrix and record one overall reputation
            # score per agent for analysis.
            reputations = self.reputation_model.update(
                observed_messages=reputation_messages,
                rewards=rewards,
                choices=choices,
            )

            # Current outcome messages become the previous messages observed in
            # the next timestep.
            self.previous_messages = outcome_messages

            # Save timestep-level logs.
            self.choices_log.append(list(choices))
            self.rewards_log.append(list(rewards))
            self.reputation_log.append(list(reputations))
            self.pairwise_trust_log.append(self.reputation_model.current_pairwise_trust())
            self.lying_log.append([int(message["lied"]) for message in decision_messages])

        # Convert logs into summary and timestep metrics used by the experiments.
        arm_means = [arm.mean for arm in self.config.arms]
        summary_metrics, timestep_metrics = compute_summary_metrics(
            self.choices_log,
            self.rewards_log,
            self.cumulative_rewards,
            arm_means,
            self.config.collision_policy,
        )

        # Extra social-trading metrics that are not part of the base bandit logs.
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
