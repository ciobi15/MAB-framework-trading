def main(*args, **kwargs):
    from multi_agent_bandits.experiments.thesis_social_trading.run_all import (
        main as run_all_main,
    )

    return run_all_main(*args, **kwargs)

__all__ = ["main"]
