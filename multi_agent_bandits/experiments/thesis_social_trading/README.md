# Thesis Social Trading Experiments

This folder contains dedicated experiment runners for the thesis questions:

- `sq1_communication_returns.py`
  Studies how communication type, noise, and local network structure affect
  average cumulative returns and return dispersion.
- `sq2_reputation_coordination.py`
  Studies how the reputation mechanism changes coordination outcomes over time,
  using herding, diversification, and malicious-agent robustness metrics.
- `sq2_reputation_learning_rate_sweep.py`
  Sweeps the reputation learning rate to test whether conservative or reactive
  trust updates work better under clean and malicious communication.
- `run_all.py`
  Runs the two thesis experiment sets.

Outputs are written to `results/thesis_social_trading/` by default.

Each experiment output folder includes:

- `sweep_summary.csv`: one row per scenario and seed, including reputation parameters such as strength and learning rate.
- `sweep_timestep_metrics.csv`: one row per scenario, seed, and timestep with aggregate coordination metrics.
- `sweep_agent_timestep_details.csv`: one row per scenario, seed, timestep, and agent with choices, rewards, cumulative rewards, reputation, lie flags, and malicious-agent flags.

Run the reputation learning-rate sweep with:

```bash
mab run thesis_social_trading.sq2_reputation_learning_rate_sweep --steps 4000 --save multi_agent_bandits/experiments/thesis_social_trading/results/thesis_social_trading
```
