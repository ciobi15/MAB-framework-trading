import math


class ReputationModel:
    """
    Pairwise trust model.

    Each agent maintains its own trust score for every other agent. That trust
    is updated from how well the sender's reported reward signal matches what
    eventually happens on the recommended arm, with larger updates when the
    receiver actually followed that signal.
    """

    def __init__(
        self,
        n_agents,
        enabled=False,
        strength=1.0,
        learning_rate=0.25,
        default_trust=1.0,
    ):
        self.n_agents = n_agents
        self.enabled = enabled
        self.strength = strength
        self.learning_rate = learning_rate
        self.default_trust = default_trust
        self.trust_matrix = self._fresh_trust_matrix()

    def _fresh_trust_matrix(self):
        return [
            [1.0 if receiver_idx == sender_idx else self.default_trust for sender_idx in range(self.n_agents)]
            for receiver_idx in range(self.n_agents)
        ]

    def trust(self, receiver_idx, sender_idx):
        if receiver_idx == sender_idx or not self.enabled:
            return 1.0
        return self.trust_matrix[receiver_idx][sender_idx]

    def influence_weight(self, receiver_idx, sender_idx):
        if receiver_idx == sender_idx or not self.enabled:
            return 1.0
        return self.trust_matrix[receiver_idx][sender_idx] ** self.strength

    def agent_reputation(self, agent_idx):
        if not self.enabled:
            return 1.0

        received_trust = [
            self.trust_matrix[receiver_idx][agent_idx]
            for receiver_idx in range(self.n_agents)
            if receiver_idx != agent_idx
        ]
        if not received_trust:
            return 1.0
        return sum(received_trust) / len(received_trust)

    def current_scores(self):
        return [self.agent_reputation(agent_idx) for agent_idx in range(self.n_agents)]

    def current_pairwise_trust(self):
        return [list(row) for row in self.trust_matrix]

    def update(self, observed_messages, rewards, choices):
        if not self.enabled:
            self.trust_matrix = self._fresh_trust_matrix()
            return self.current_scores()

        arm_rewards = {}
        arm_counts = {}
        for agent_idx, arm_idx in enumerate(choices):
            arm_rewards.setdefault(arm_idx, 0.0)
            arm_counts.setdefault(arm_idx, 0)
            arm_rewards[arm_idx] += rewards[agent_idx]
            arm_counts[arm_idx] += 1

        reward_scale = max(1.0, max(abs(reward) for reward in rewards))

        for receiver_idx, messages in enumerate(observed_messages):
            for message in messages:
                sender_idx = message["sender_id"]
                if sender_idx == receiver_idx:
                    continue

                arm_idx = message["arm"]
                if message.get("message_type") == "preview":
                    reference_reward = arm_rewards[arm_idx] / arm_counts[arm_idx]
                else:
                    reference_reward = message["truthful_value"]

                prediction_error = abs(message["reported_value"] - reference_reward)
                accuracy_signal = math.exp(-prediction_error / reward_scale)

                followed_sender = choices[receiver_idx] == arm_idx
                update_strength = 1.0 if followed_sender else 0.35
                target_trust = 0.25 + 1.75 * accuracy_signal

                current_trust = self.trust_matrix[receiver_idx][sender_idx]
                effective_rate = self.learning_rate * update_strength
                self.trust_matrix[receiver_idx][sender_idx] = (
                    (1.0 - effective_rate) * current_trust
                    + effective_rate * target_trust
                )

            self.trust_matrix[receiver_idx][receiver_idx] = 1.0

        return self.current_scores()
