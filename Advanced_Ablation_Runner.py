import os
import pandas as pd
from tools.logs import Logger
from trainer import DRLTrainer, CustomCallback
from Ablation_Runner import DataAnalyzer, DataVisualizer


def run_drl_with_config(algo, env_name, env_config, total_timesteps, seed, reward_config):
    import numpy as np
    import torch
    import random
    import time

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    callback = CustomCallback(verbose=0)
    trainer = DRLTrainer(
        algo=algo,
        env_name=env_name,
        env_config=env_config,
        learning_rate=1e-4
    )
    trainer.model.policy_kwargs = dict(
        reward_config=reward_config
    )

    start_time = time.time()
    trainer.train(total_timestamps=total_timesteps, callback=callback)
    end_time = time.time()

    record = {
        "episodes": callback.num_episodes,
        "success_rate": callback.successful_rate,
        "avg_cost": callback.avg_cost,
        "min_cost": callback.min_cost,
        "runtime_sec": round(end_time - start_time, 2),
        "sample_efficiency": round(callback.num_success / total_timesteps, 6) if total_timesteps > 0 else 0
    }
    return record


def analyze_and_plot(csv_path, stats_path):
    analyzer = DataAnalyzer(csv_path)
    stats_df = analyzer.calculate_stats(
        metrics=["success_rate", "avg_cost", "min_cost", "runtime_sec", "sample_efficiency"]
    )
    analyzer.save_stats(stats_path)
    plot_path = stats_path.replace(".csv", ".png")
    DataVisualizer.plot_stats(stats_df, plot_path)


class AdvancedAblationRunner:
    def __init__(self, env_name, env_config, seeds, timesteps, save_dir="results/advanced_ablation_testReward"):
        self.env_name = env_name
        self.env_config = env_config
        self.seeds = seeds
        self.timesteps = timesteps
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.experiments = {
            "Full": (True, True, True, True, True),
            "NoTarget": (True, True, False, True, True),
            "NoEllipseTerm": (True, True, True, False, True),
            "NoCost": (True, True, True, True, False),
            "OnlyTarget": (True, True, True, False, False),
            "OnlyEllipse": (True, True, False, True, False),
            "OnlyCost": (True, True, False, False, True),
            "None": (True, True, False, False, False),
            "NoEllipseGuidance": (True, False, True, False, True),
            "NoVerification": (False, False, True, False, True),
            "RandomSearch": (False, False, False, False, False),
        }

    def run(self):
        for mode, (use_verif, use_ellipse, use_target, use_ellipse_term, use_cost) in self.experiments.items():
            env_config = self.env_config
            if not use_verif:
                if "In2-dr" in self.env_config:
                    env_config = "In2-dr_origin"
                else:
                    continue

            csv_path = f"{self.save_dir}/results_{env_config}_{mode}.csv"
            if os.path.exists(csv_path):
                print(f"[Skip] Already exists: {csv_path}")
                continue

            print(f"\n>>> Running mode: {mode} | config: {env_config}")
            records = []
            for seed in self.seeds:
                record = run_drl_with_config(
                    algo="PPO",
                    env_name=self.env_name,
                    env_config=env_config,
                    total_timesteps=self.timesteps,
                    seed=seed,
                    reward_config={
                        "use_target": use_target,
                        "use_ellipse": use_ellipse_term,
                        "use_cost": use_cost
                    }
                )
                record["seed"] = seed
                records.append(record)

            df = pd.DataFrame(records)
            df.to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")

            analyze_and_plot(csv_path, f"{self.save_dir}/summary_{env_config}_{mode}.csv")


if __name__ == "__main__":
    runner = AdvancedAblationRunner(
        env_name="ADAS",
        env_config="In2-dr",
        seeds=list(range(1, 21)),
        timesteps=500_000
    )
    runner.run()
