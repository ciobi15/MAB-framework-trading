# Social Trading Extension

This document describes the social-trading extension added on top of the existing multi-agent bandit framework.

## Design goal

The original framework is preserved as the stable base.

The extension adds a separate research layer for retail social trading without rewriting the current core classes.
This keeps the original environment, strategies, and experiment workflow intact while adding new files for communication, networks, reputation, metrics, plotting, and sweeps.

## What stays reused from the original framework

- `multi_agent_bandits.core.arm.Arm`
- reward-sharing policies from `multi_agent_bandits.core.reward_sharing`
- the experiment-script pattern in `multi_agent_bandits.experiments`
- the existing CLI command `mab run ...`

## New files

- `multi_agent_bandits/social_trading/config_social_trading.py`
- `multi_agent_bandits/social_trading/social_network.py`
- `multi_agent_bandits/social_trading/reputation.py`
- `multi_agent_bandits/social_trading/social_agent.py`
- `multi_agent_bandits/social_trading/social_metrics.py`
- `multi_agent_bandits/social_trading/plots_social_trading.py`
- `multi_agent_bandits/social_trading/multi_agent_simulation.py`
- `multi_agent_bandits/social_trading/sweep_runner.py`
- `multi_agent_bandits/experiments/social_trading.py`
- `multi_agent_bandits/experiments/social_trading_sweep.py`

## Architecture

### `SocialTradingSimulation`

This is the main research simulation loop.

At each timestep:

1. Each agent receives social information from the previous timestep.
2. Social information depends on the communication structure:
   - `none`
   - `global`
   - `local`
3. Under `local`, neighbors are determined by a network topology:
   - `fully_connected`
   - `random_graph`
   - `ring_lattice`
4. Each agent chooses an arm using its own estimates plus the social signal.
5. Rewards are sampled from the original `Arm` objects.
6. Collision rewards are resolved using the existing collision policy interface.
7. Reputations are updated from cumulative average performance.
8. Results are logged for later analysis.

### `SocialTradingAgent`

This is a simple epsilon-greedy agent with additive social influence.

Own estimate for an arm:

- learned from personal reward history

Social component for an arm:

- built from communicated rewards about that arm
- optionally weighted by sender reputation
- scaled by `social_influence_strength`

This is intentionally transparent and easy to explain in a thesis.

### `ReputationModel`

Reputation is optional.

When enabled:

- each agent starts with score `1.0`
- scores are updated from cumulative average returns
- relative performance is normalized into the range `[1, 2]`
- influence weights are computed as `reputation_score ** reputation_strength`

This creates a tunable but interpretable reputation mechanism.

## Communication noise

Communication noise is implemented as Gaussian noise added to the reported reward that one agent shares with another.

Higher noise means social information becomes less reliable.

## Metrics

The extension computes:

### Performance

- average cumulative return across agents
- average reward over time
- final group performance
- expected regret when the exact oracle is computationally manageable

### Inequality

- standard deviation of cumulative returns
- variance of cumulative returns
- Gini coefficient

### Coordination

- arm-choice distribution over time
- concentration on the most popular arm
- entropy of the arm-choice distribution
- number of distinct arms chosen per timestep

These are designed to help detect herding versus diversification.

## Outputs

Single-run experiments save:

- `choices.csv`
- `rewards.csv`
- `reputations.csv`
- `summary_metrics.csv`
- `timestep_metrics.csv`
- `network.csv`
- plot files when plotting is enabled

Sweep experiments save:

- `sweep_summary.csv`
- `sweep_timestep_metrics.csv`
- `sweep_tradeoff.png`

## How to run

### Single experiment

```bash
mab run social_trading --steps 1000 --save results/social_trading --plot-rewards
```

### Parameter sweep

```bash
mab run social_trading_sweep --steps 400 --save results/social_trading_sweep --plot-rewards
```

## Assumptions

- agents communicate information from the previous timestep, not the current one
- communication shares chosen arm and realized reward
- reputation is based on cumulative average performance
- local communication uses an undirected static network during a run
- the extension is additive and does not replace the original framework

## How this helps answer the research questions

The saved outputs make it possible to compare:

- communication structures (`none`, `global`, `local`)
- network topologies under local communication
- different communication noise levels
- reputation on versus off

These outputs support the main thesis questions by linking:

- collective performance metrics
- inequality metrics
- coordination metrics that capture herding and diversification

In particular, `sweep_summary.csv` can be used to study the trade-off between high returns and low inequality, while `timestep_metrics.csv` helps inspect whether reputation and communication increase coordination over time.
