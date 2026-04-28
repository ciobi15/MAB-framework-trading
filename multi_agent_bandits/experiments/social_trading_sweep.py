from multi_agent_bandits.core.arm import Arm
from multi_agent_bandits.core.reward_sharing import linear_share
from multi_agent_bandits.social_trading.config_social_trading import SocialTradingConfig
from multi_agent_bandits.social_trading.sweep_runner import SweepRunner


def main(steps=400, save_dir=None, plot_rewards=False, plot_frequencies=False):
    base_config = SocialTradingConfig(
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

    thesis_scenarios = [
        {
            "scenario": "pure_ucb",
            "communication_structure": "none",
            "use_reputation": False,
            "communication_noise": 0.0,
        },
        {
            "scenario": "ucb_social_information",
            "communication_structure": "global",
            "use_reputation": False,
            "communication_noise": 0.0,
        },
        {
            "scenario": "ucb_social_reputation",
            "communication_structure": "global",
            "use_reputation": True,
            "communication_noise": 0.0,
            "reputation_strength": 1.0,
            "malicious_agent_ratio": 0.0,
            "lying_probability": 0.0,
            "lie_magnitude": 0.0,
        },
        {
            "scenario": "ucb_social_reputation_deceptive",
            "communication_structure": "global",
            "use_reputation": True,
            "communication_noise": 0.0,
            "reputation_strength": 1.0,
            "malicious_agent_ratio": 0.25,
            "lying_probability": 0.7,
            "lie_magnitude": 1.0,
        },
    ]

    runner = SweepRunner(
        base_config=base_config,
        sweep_parameters={},
        seeds=[1, 7, 21],
        output_dir=save_dir or "results/social_trading_sweep",
        scenario_rows=thesis_scenarios,
    )
    summary_rows, _ = runner.run(save_plots=plot_rewards or plot_frequencies or True)

    print("Social trading thesis scenario summary")
    print(f"Runs completed: {len(summary_rows)}")
