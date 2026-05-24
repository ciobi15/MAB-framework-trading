import math


class ReputationModel:
    """
    Pairwise trust model.

    Each agent maintains its own trust score for every other agent. That trust
    is updated from how well the sender's reported reward signal matches what
    eventually happens on the recommended arm, with larger updates when the
    receiver actually followed that signal.

    This is pairwise rather than global: agent A can trust agent B more than
    agent C does. The public "reputation" of an agent is computed afterwards as
    the average trust it receives from the other agents.
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

        # When disabled, the rest of the simulation can still call this object,
        # but every trust and influence lookup behaves as neutral trust = 1.0.
        self.enabled = enabled
        self.strength = strength
        self.learning_rate = learning_rate
        self.default_trust = default_trust
        self.trust_matrix = self._fresh_trust_matrix()

    def _fresh_trust_matrix(self):
        """
        Build receiver-by-sender trust scores.

        Rows are receivers, columns are senders. The diagonal is always 1.0
        because an agent's trust in itself is treated as neutral/perfect.
        """
        return [
            [1.0 if receiver_idx == sender_idx else self.default_trust for sender_idx in range(self.n_agents)]
            for receiver_idx in range(self.n_agents)
        ]

    def trust(self, receiver_idx, sender_idx):
        """Return how much receiver_idx currently trusts sender_idx."""
        if receiver_idx == sender_idx or not self.enabled:
            return 1.0
        return self.trust_matrix[receiver_idx][sender_idx]

    def influence_weight(self, receiver_idx, sender_idx):
        """
        Convert trust into a weight for social influence.

        strength controls how sharply trust matters. A strength of 1.0 uses the
        trust score directly; larger values amplify differences between low and
        high trust.
        """
        if receiver_idx == sender_idx or not self.enabled:
            return 1.0
        return self.trust_matrix[receiver_idx][sender_idx] ** self.strength

    def agent_reputation(self, agent_idx):
        """Summarize one sender's reputation as average trust from others."""
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
        """Return one reputation score per agent for logging/plots."""
        return [self.agent_reputation(agent_idx) for agent_idx in range(self.n_agents)]

    def current_pairwise_trust(self):
        """Return a copy of the full trust matrix for detailed analysis."""
        return [list(row) for row in self.trust_matrix]

    def update(self, observed_messages, rewards, choices):
        """
        Update pairwise trust after one simulation step.

        observed_messages contains the recommendations each receiver saw.
        rewards and choices describe what actually happened after all agents
        acted, which lets the model compare reported claims with outcomes.
        """
        if not self.enabled:
            self.trust_matrix = self._fresh_trust_matrix()
            return self.current_scores()

        # For preview messages, the best reference point is the realized average
        # reward on the recommended arm among agents who actually chose it.
        arm_rewards = {}
        arm_counts = {}
        for agent_idx, arm_idx in enumerate(choices):
            arm_rewards.setdefault(arm_idx, 0.0)
            arm_counts.setdefault(arm_idx, 0)
            arm_rewards[arm_idx] += rewards[agent_idx]
            arm_counts[arm_idx] += 1

        # Scale prediction errors so that the exponential accuracy score behaves
        # sensibly across reward distributions with different magnitudes.
        reward_scale = max(1.0, max(abs(reward) for reward in rewards))

        for receiver_idx, messages in enumerate(observed_messages):
            for message in messages:
                sender_idx = message["sender_id"]
                if sender_idx == receiver_idx:
                    continue

                arm_idx = message["arm"]

                if message.get("message_type") == "preview":
                    reference_reward = arm_rewards[arm_idx] / arm_counts[arm_idx]
                elif choices[receiver_idx] == arm_idx:
                    # Outcome messages are treated as advice about the arm's
                    # next payoff. If the receiver followed that advice, judge
                    # it against the receiver's own realized reward.
                    reference_reward = rewards[receiver_idx]
                elif arm_idx in arm_rewards:
                    # If the receiver did not follow the outcome message but
                    # other agents tried that arm this step, use their realized
                    # average as weaker evidence about the sender's accuracy.
                    reference_reward = arm_rewards[arm_idx] / arm_counts[arm_idx]
                else:
                    # If nobody tried that arm this step, there is no new
                    # outcome evidence. Fall back to the sender's stored truth.
                    reference_reward = message["truthful_value"]

                prediction_error = abs(message["reported_value"] - reference_reward)
                accuracy_signal = math.exp(-prediction_error / reward_scale)

                # Following a sender is stronger evidence: if the receiver acted
                # on the advice, the sender's accuracy matters more for trust.
                followed_sender = choices[receiver_idx] == arm_idx
                update_strength = 1.0 if followed_sender else 0.35

                # Map accuracy into the trust range. Very inaccurate messages
                # approach 0.25, while highly accurate messages approach 2.0.
                target_trust = 0.25 + 1.75 * accuracy_signal

                current_trust = self.trust_matrix[receiver_idx][sender_idx]
                effective_rate = self.learning_rate * update_strength

                # Smoothly move current trust toward the target instead of
                # replacing it, so one unusual outcome does not dominate history.
                self.trust_matrix[receiver_idx][sender_idx] = (
                    (1.0 - effective_rate) * current_trust
                    + effective_rate * target_trust
                )

            # Keep the self-trust diagonal fixed after each update cycle.
            self.trust_matrix[receiver_idx][receiver_idx] = 1.0

        return self.current_scores()
