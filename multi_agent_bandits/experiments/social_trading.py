from multi_agent_bandits.core.arm import Arm
from multi_agent_bandits.core.reward_sharing import linear_share
from multi_agent_bandits.social_trading.config_social_trading import SocialTradingConfig
from multi_agent_bandits.social_trading.multi_agent_simulation import (
    SocialTradingSimulation,
)


def main(steps=1000, save_dir=None, plot_rewards=False, plot_frequencies=False):
    config = SocialTradingConfig(
        n_agents=8,
        arms=[
            Arm(mean=0.8, sd=1.0),
            Arm(mean=1.1, sd=1.0),
            Arm(mean=1.5, sd=1.0),
            Arm(mean=1.9, sd=1.0),
        ],
        timesteps=steps,
        collision_policy=linear_share,
        communication_structure="global",
        network_topology="fully_connected",
        communication_noise=0.2,
        use_reputation=True,
        reputation_strength=1.0,
        social_influence_strength=0.6,
        ucb_exploration=2.0,
        crowding_penalty=0.3,
        malicious_agent_ratio=0.25,
        lying_probability=0.7,
        lie_magnitude=1.0,
        save_dir=save_dir or "results/social_trading",
    )

    simulation = SocialTradingSimulation(config)
    result = simulation.run(
        save_outputs=True,
        save_plots=plot_rewards or plot_frequencies,
    )

    print("Social trading summary")
    for key, value in result.summary_metrics.items():
        print(f"{key}: {value}")
