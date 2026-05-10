# Social Trading Experiments: Results Presentation

Short presentation draft for supervisor discussion.

## 1. Evaluation Goal

- The experiments evaluate a social multi-agent bandit framework.
- Agents learn with UCB, but can also use social information from other agents.
- The final evaluation now focuses on two sub-questions:
  - **SQ1:** How do communication type, noise level, and network structure affect average and spread of cumulative returns?
  - **SQ2:** How do reputation mechanisms change arm-choice distributions over time, especially herding versus diversification, under clean and malicious-agent communication?

## 2. Experimental Setup

- Each experiment uses 8 agents and 12 arms.
- Each scenario is repeated over 5 random seeds:
  - `1, 7, 21, 42, 84`
- Each run uses 4000 timesteps.
- Results are aggregated across seeds using means and standard deviations.
- Communication uses two message types:
  - **Preview messages:** sent before rewards are known; show what an agent is currently choosing and estimating.
  - **Outcome messages:** sent after rewards are known; show what reward an agent reports receiving.
- Reputation, when enabled, weights messages by pairwise trust:
  - trusted senders influence more,
  - less trusted senders influence less.
- Malicious agents, when enabled in SQ2, make up 25% of agents, lie with probability `0.7`, and use lie magnitude `1.0`.

## 3. SQ1: Communication And Returns

### Research Question

- **SQ1:** How do communication type, noise level, and network structure affect average and spread of cumulative returns across agents?

### What SQ1 Compares

- Clean global communication:
  - global communication,
  - fully connected,
  - no noise.
- Noisy global communication:
  - noise levels `0.15`, `0.3`, and `0.6`.
- Local communication:
  - `ring_lattice`,
  - `random_graph`,
  - `fully_connected`.
- Local communication noise:
  - `0.0`,
  - `0.15`,
  - `0.3`,
  - `0.6`.
- Reputation and malicious agents are disabled in SQ1.

### SQ1 Candidate Plots

- `sq1_returns_vs_noise.png`
- `sq1_local_heatmaps.png`
- `sq1_expected_regret_vs_noise.png`
- `sq1_return_spread_vs_noise.png`
- `sq1_gini_vs_noise.png`
- `sq1_entropy_vs_noise.png`
- `sq1_herding_vs_noise.png`
- `sq1_distinct_arms_vs_noise.png`
- `sq1_return_regret_tradeoff.png`
- `sq1_performance_inequality_tradeoff.png`
- `sq1_ranked_returns.png`
- `sq1_ranked_regret.png`
- `sq1_noise_penalty_by_topology.png`

### Main SQ1 Results To Check

- Global and fully connected local communication should produce the strongest returns.
- Sparse local communication should reduce performance because agents have less access to useful information.
- High communication noise should reduce returns and increase regret.
- Ring-lattice communication may produce lower returns but more equal outcomes.
- The best paper figures are likely:
  - returns versus noise,
  - local heatmaps,
  - performance-inequality tradeoff,
  - expected regret versus noise.

## 4. SQ2: Reputation, Coordination, And Malicious Agents

### Research Question

- **SQ2:** How do reputation mechanisms change the distribution of arm choices across agents over time, and do they help when some agents are malicious?

### What SQ2 Compares

- Global communication, clean, without reputation.
- Global communication, clean, with reputation.
- Global communication, malicious agents, without reputation.
- Global communication, malicious agents, with reputation.
- Local ring-lattice communication, clean, without reputation.
- Local ring-lattice communication, clean, with reputation.
- Local ring-lattice communication, malicious agents, without reputation.
- Local ring-lattice communication, malicious agents, with reputation.

### SQ2 Candidate Plots

- `sq2_coordination_trajectories.png`
- `sq2_herding_diversification_map.png`
- `sq2_group_reward_trajectories.png`
- `sq2_choice_entropy_trajectories.png`
- `sq2_herding_trajectories.png`
- `sq2_distinct_arms_trajectories.png`
- `sq2_group_reward_clean_vs_malicious.png`
- `sq2_choice_entropy_clean_vs_malicious.png`
- `sq2_average_cumulative_return_bars.png`
- `sq2_expected_regret_bars.png`
- `sq2_gini_bars.png`
- `sq2_final_entropy_bars.png`
- `sq2_final_herding_bars.png`
- `sq2_mean_lie_rate_bars.png`
- `sq2_final_reputation_bars.png`
- `sq2_malicious_return_penalty.png`
- `sq2_reputation_return_gain.png`

### Main SQ2 Results To Check

- Reputation should be evaluated separately in clean and malicious settings.
- The clean setting shows whether reputation changes coordination when messages are honest.
- The malicious setting shows whether reputation protects the group from deceptive social information.
- The key metrics are:
  - average cumulative return,
  - expected regret,
  - most popular arm share,
  - choice entropy,
  - distinct arms chosen,
  - mean lie rate,
  - final mean reputation,
  - Gini coefficient.
- The best paper figures are likely:
  - coordination trajectories,
  - clean versus malicious reward trajectories,
  - malicious return penalty,
  - reputation return gain,
  - herding-diversification map.

## 5. Overall Findings To Frame

- Communication improves learning when information can spread broadly.
- Network structure matters because it controls how quickly useful information moves between agents.
- Communication noise is useful for testing whether social learning remains stable under imperfect information.
- Reputation should be presented as both a coordination mechanism and a robustness mechanism.
- Malicious-agent conditions make SQ2 stronger because they test whether reputation is useful when social information can be intentionally misleading.

## 6. Notes For Supervisor Discussion

- Be careful not to overclaim small numerical differences, especially where results are close.
- Emphasize that results are averaged across 5 seeds.
- Explain that SQ3 was removed, and the performance-inequality analysis is now represented through SQ1/SQ2 metrics and plots instead of a separate experiment.
- Choose only the clearest 4-6 figures for the paper; the generated plot gallery is meant for selection, not automatic inclusion.
