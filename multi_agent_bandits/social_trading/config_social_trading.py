from dataclasses import dataclass, field

from multi_agent_bandits.core.arm import Arm
from multi_agent_bandits.core.reward_sharing import linear_share


@dataclass
class SocialTradingConfig:
    """
    Configuration for the social-trading extension.
    """

    n_agents: int = 8
    arms: list = field(
        default_factory=lambda: [
            Arm(mean=1.0, sd=1.0),
            Arm(mean=1.4, sd=1.0),
            Arm(mean=1.8, sd=1.0),
            Arm(mean=2.2, sd=1.0),
        ]
    )
    timesteps: int = 1000
    seed: int | None = None
    collision_policy: callable = linear_share
    communication_structure: str = "none"
    network_topology: str = "fully_connected"
    random_edge_prob: float = 0.35
    ring_neighbors: int = 1
    communication_noise: float = 0.0
    use_reputation: bool = False
    reputation_strength: float = 1.0
    social_influence_strength: float = 0.5
    ucb_exploration: float = 2.0
    crowding_penalty: float = 0.25
    malicious_agent_ratio: float = 0.0
    lying_probability: float = 0.75
    lie_magnitude: float = 1.0
    save_dir: str | None = None

    @property
    def n_arms(self):
        return len(self.arms)

    def validate(self):
        valid_structures = {"none", "global", "local"}
        valid_topologies = {"fully_connected", "random_graph", "ring_lattice"}

        if self.n_agents <= 0:
            raise ValueError("n_agents must be positive.")
        if self.n_arms <= 0:
            raise ValueError("At least one arm is required.")
        if self.timesteps <= 0:
            raise ValueError("timesteps must be positive.")
        if self.communication_structure not in valid_structures:
            raise ValueError(
                f"communication_structure must be one of {sorted(valid_structures)}."
            )
        if self.network_topology not in valid_topologies:
            raise ValueError(
                f"network_topology must be one of {sorted(valid_topologies)}."
            )
        if not 0.0 <= self.random_edge_prob <= 1.0:
            raise ValueError("random_edge_prob must be between 0 and 1.")
        if self.ring_neighbors < 0:
            raise ValueError("ring_neighbors must be non-negative.")
        if self.communication_noise < 0:
            raise ValueError("communication_noise must be non-negative.")
        if self.reputation_strength < 0:
            raise ValueError("reputation_strength must be non-negative.")
        if self.social_influence_strength < 0:
            raise ValueError("social_influence_strength must be non-negative.")
        if self.ucb_exploration < 0:
            raise ValueError("ucb_exploration must be non-negative.")
        if self.crowding_penalty < 0:
            raise ValueError("crowding_penalty must be non-negative.")
        if not 0.0 <= self.malicious_agent_ratio <= 1.0:
            raise ValueError("malicious_agent_ratio must be between 0 and 1.")
        if not 0.0 <= self.lying_probability <= 1.0:
            raise ValueError("lying_probability must be between 0 and 1.")
        if self.lie_magnitude < 0:
            raise ValueError("lie_magnitude must be non-negative.")
