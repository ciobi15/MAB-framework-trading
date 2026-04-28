# Thesis Social Trading Experiments

This folder contains dedicated experiment runners for the thesis questions:

- `sq1_communication_returns.py`
  Studies how communication type, noise, and local network structure affect
  average cumulative returns and return dispersion.
- `sq2_reputation_coordination.py`
  Studies how the reputation mechanism changes coordination outcomes over time,
  using herding and diversification metrics.
- `sq3_tradeoff.py`
  Compares communication and reputation conditions on the
  performance-inequality trade-off.
- `run_all.py`
  Runs all three experiment sets.

Outputs are written to `results/thesis_social_trading/` by default.
