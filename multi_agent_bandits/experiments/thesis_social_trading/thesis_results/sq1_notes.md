# SQ1 Notes

- Communication type and network structure have a large effect on average cumulative returns.
- The global baseline reaches `246.04` average cumulative return with return spread `27.61`.
- In the global setting, noise has only a modest negative effect.
- Average cumulative return moves from `246.04` at noise `0.00` to `242.67` at noise `0.60`.
- Expected regret increases from `151.69` to `178.64` as global noise rises.
- The strongest-performing local condition is the fully connected local network, which mirrors the global case.
- The best of these settings reaches `246.57` at noise `0.15`.
- The ring-lattice local network performs much worse in return terms.
- Its average cumulative return ranges from `198.42` to `202.12` across the tested noise levels.
- The random-graph local network performs slightly better than the ring lattice in returns but is much less equal.
- Its average cumulative return stays around `201.85-204.94`.
- Its cross-agent return spread is the highest among the local settings, peaking at `43.84` at noise `0.30`.
- The ring-lattice setting is more equal than the random graph setting.
- Ring-lattice Gini values stay around `0.038-0.048`, while random-graph Gini values are around `0.090-0.110`.
- A thesis sentence you can use:
- SQ1 shows that communication quality matters as much as communication presence: globally shared information is robust to moderate noise, while sparse or irregular local networks reduce returns and can substantially widen the spread of outcomes across agents.
