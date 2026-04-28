import math
import random

from multi_agent_bandits.core.agent import Agent


class SocialTradingAgent(Agent):
    """
    UCB learner with additive social information and optional deceptive
    reward reports.
    """

    def __init__(
        self,
        n_arms,
        ucb_exploration=2.0,
        social_influence_strength=0.5,
        crowding_penalty=0.25,
        lying_probability=0.0,
        lie_magnitude=0.0,
        name=None,
    ):
        super().__init__(n_arms, name=name)
        self.ucb_exploration = ucb_exploration
        self.social_influence_strength = social_influence_strength
        self.crowding_penalty = crowding_penalty
        self.lying_probability = lying_probability
        self.lie_magnitude = lie_magnitude
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        self.total_steps = 0
        self.last_arm = None
        self.action_history = []
        self.reward_history = []
        self.cumulative_reward = 0.0

    def _normalize_social_inputs(self, social_signal=None, observed_counts=None):
        social_signal = social_signal or [0.0] * self.n_arms
        observed_counts = observed_counts or [0] * self.n_arms
        return social_signal, observed_counts

    def estimate_payoff(self, arm_idx, social_signal=None, observed_counts=None):
        social_signal, observed_counts = self._normalize_social_inputs(
            social_signal=social_signal,
            observed_counts=observed_counts,
        )

        estimate = self.values[arm_idx]
        estimate += self.social_influence_strength * social_signal[arm_idx]
        estimate -= self.crowding_penalty * observed_counts[arm_idx]
        return estimate

    def choose_arm(self, social_signal=None, observed_counts=None):
        social_signal, observed_counts = self._normalize_social_inputs(
            social_signal=social_signal,
            observed_counts=observed_counts,
        )
        self.total_steps += 1
        warm_start_bonus = math.sqrt(
            self.ucb_exploration * math.log(self.total_steps + 1)
        )

        scores = []
        for arm_idx in range(self.n_arms):
            if self.counts[arm_idx] == 0:
                score = self.estimate_payoff(
                    arm_idx,
                    social_signal=social_signal,
                    observed_counts=observed_counts,
                )
                score += warm_start_bonus
            else:
                bonus = math.sqrt(
                    (self.ucb_exploration * math.log(self.total_steps + 1))
                    / self.counts[arm_idx]
                )
                score = self.estimate_payoff(
                    arm_idx,
                    social_signal=social_signal,
                    observed_counts=observed_counts,
                )
                score += bonus
            scores.append(score)

        self.last_arm = max(range(self.n_arms), key=lambda arm_idx: scores[arm_idx])
        return self.last_arm

    def generate_signal(self, arm_idx, social_signal=None, observed_counts=None):
        truthful_value = self.estimate_payoff(
            arm_idx,
            social_signal=social_signal,
            observed_counts=observed_counts,
        )

        lied = random.random() < self.lying_probability
        distortion = self.lie_magnitude if lied else 0.0

        return {
            "truthful_value": truthful_value,
            "reported_value": truthful_value + distortion,
            "lied": lied,
            "distortion": distortion,
        }

    def update(self, reward, chosen_arm=None):
        arm = self.last_arm if chosen_arm is None else chosen_arm
        self.counts[arm] += 1
        step = 1 / self.counts[arm]
        self.values[arm] += step * (reward - self.values[arm])

        self.action_history.append(arm)
        self.reward_history.append(reward)
        self.cumulative_reward += reward
