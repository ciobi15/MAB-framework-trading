# Main RQ Notes

- The final thesis runs use `8` agents, `12` arms, `4000` timesteps, and seeds `1, 7, 21, 42, 84`.
- Communication structure clearly shapes collective investor performance.
- In SQ1, global and fully connected local communication produce the highest returns, with clean global/fully connected conditions reaching about `3056.10` average cumulative return.
- High communication noise reduces performance; global noise `0.6` falls below the clean global condition.
- SQ2 shows that malicious agents reduce performance in both global and local communication.
- In global malicious communication, reputation improves average cumulative return from `2787.33` to `2832.76`.
- In local malicious communication, reputation improves average cumulative return from `2300.52` to `2446.28`.
- Reputation also improves diversification in both malicious settings by increasing entropy and reducing the most-popular-arm share.
- Under clean local communication, reputation is not automatically beneficial and lowers average cumulative return from `2851.82` to `2810.73`.
- A compact thesis sentence you can use:
- In the multi-agent retail social trading model, communication improves collective learning when information can spread broadly, while reputation is most valuable as a robustness mechanism when social information can be deceptive.
