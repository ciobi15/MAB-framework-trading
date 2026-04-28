import copy
import csv
import itertools
import os

from multi_agent_bandits.social_trading.multi_agent_simulation import SocialTradingSimulation
from multi_agent_bandits.social_trading.plots_social_trading import plot_sweep_summary


class SweepRunner:
    """
    Run parameter sweeps or explicit named scenarios over communication,
    reputation, and deception settings.
    """

    def __init__(
        self,
        base_config,
        sweep_parameters,
        seeds,
        output_dir,
        scenario_rows=None,
    ):
        self.base_config = base_config
        self.sweep_parameters = sweep_parameters
        self.seeds = list(seeds)
        self.output_dir = output_dir
        self.scenario_rows = list(scenario_rows or [])
        os.makedirs(self.output_dir, exist_ok=True)

    def _parameter_rows(self):
        if self.scenario_rows:
            for row in self.scenario_rows:
                yield self._canonicalize_parameters(row)
            return

        parameter_names = list(self.sweep_parameters.keys())
        parameter_values = [self.sweep_parameters[name] for name in parameter_names]

        for combination in itertools.product(*parameter_values):
            row = dict(zip(parameter_names, combination))
            yield self._canonicalize_parameters(row)

    def _canonicalize_parameters(self, row):
        canonical = dict(row)

        if canonical.get("communication_structure") != "local":
            canonical["network_topology"] = "fully_connected"

        if not canonical.get("use_reputation", self.base_config.use_reputation):
            canonical["reputation_strength"] = 0.0

        communication_structure = canonical.get(
            "communication_structure",
            self.base_config.communication_structure,
        )
        use_reputation = canonical.get(
            "use_reputation",
            self.base_config.use_reputation,
        )
        if communication_structure == "none":
            canonical["malicious_agent_ratio"] = 0.0
            canonical["lying_probability"] = 0.0
            canonical["lie_magnitude"] = 0.0

        if "scenario" not in canonical:
            canonical["scenario"] = self._scenario_name(canonical)

        return canonical

    def _scenario_name(self, row):
        communication_structure = row.get(
            "communication_structure",
            self.base_config.communication_structure,
        )
        use_reputation = row.get(
            "use_reputation",
            self.base_config.use_reputation,
        )
        malicious_agent_ratio = row.get(
            "malicious_agent_ratio",
            self.base_config.malicious_agent_ratio,
        )
        lying_probability = row.get(
            "lying_probability",
            self.base_config.lying_probability,
        )
        lie_magnitude = row.get(
            "lie_magnitude",
            self.base_config.lie_magnitude,
        )

        if communication_structure == "none":
            return "pure_ucb"
        if not use_reputation:
            return "ucb_social_information"
        if (
            malicious_agent_ratio > 0
            and lying_probability > 0
            and lie_magnitude > 0
        ):
            return "ucb_social_reputation_deceptive"
        return "ucb_social_reputation"

    def run(self, save_plots=True):
        summary_rows = []
        timestep_rows = []
        seen = set()

        for parameter_row in self._parameter_rows():
            signature = tuple(sorted(parameter_row.items()))
            if signature in seen:
                continue
            seen.add(signature)

            for seed in self.seeds:
                config = copy.deepcopy(self.base_config)
                config.seed = seed
                config.save_dir = None

                for name, value in parameter_row.items():
                    setattr(config, name, value)

                simulation = SocialTradingSimulation(config)
                result = simulation.run(save_outputs=False, save_plots=False)

                summary_row = dict(parameter_row)
                summary_row["seed"] = seed
                summary_row.update(result.summary_metrics)
                summary_rows.append(summary_row)

                for timestep_metric in result.timestep_metrics:
                    row = dict(parameter_row)
                    row["seed"] = seed
                    row.update(timestep_metric)
                    timestep_rows.append(row)

        self._write_summary(summary_rows)
        self._write_timestep_metrics(timestep_rows)

        if save_plots:
            plot_sweep_summary(summary_rows, self.output_dir)

        return summary_rows, timestep_rows

    def _write_summary(self, summary_rows):
        if not summary_rows:
            return

        path = os.path.join(self.output_dir, "sweep_summary.csv")
        with open(path, "w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    def _write_timestep_metrics(self, timestep_rows):
        if not timestep_rows:
            return

        path = os.path.join(self.output_dir, "sweep_timestep_metrics.csv")
        with open(path, "w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(timestep_rows[0].keys()))
            writer.writeheader()
            writer.writerows(timestep_rows)
