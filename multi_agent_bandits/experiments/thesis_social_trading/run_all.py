from multi_agent_bandits.experiments.thesis_social_trading.common import DEFAULT_RESULTS_ROOT
from multi_agent_bandits.experiments.thesis_social_trading.sq1_communication_returns import (
    main as run_sq1,
)
from multi_agent_bandits.experiments.thesis_social_trading.sq2_reputation_coordination import (
    main as run_sq2,
)


def main(steps=4000, seeds=None, save_dir=None):
    save_dir = save_dir or DEFAULT_RESULTS_ROOT
    run_sq1(steps=steps, seeds=seeds, save_dir=save_dir)
    run_sq2(steps=steps, seeds=seeds, save_dir=save_dir)


if __name__ == "__main__":
    main()
