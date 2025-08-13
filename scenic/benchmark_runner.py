import os
import time
import argparse
import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
import json
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from scenic.syntax.veneer import endSimulation, currentSimulation

from scenic_carla_env import ScenicCarlaEnv
from tools.logs import Logger

class CarlaScenicCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.success_count = 0
        self.total_episodes = 0
        self.total_cost = 0.0
        self.min_cost = float('inf')
        self.unsafe_counts = 0
        self.episode_lengths = []
        self.training_start_time = time.time()
        self.episode_metrics = []

    def _on_training_start(self):
        self.training_start_time = time.time()

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        done = self.locals['dones'][0]
        reward = self.locals['rewards'][0]

        if done:
            self.total_episodes += 1
            self.episode_lengths.append(self.num_timesteps - sum(self.episode_lengths))

            if info.get("is_unsafe", False):
                self.unsafe_counts += 1

            if info.get("termination_reason") == "success":
                self.success_count += 1
                if info.get("cost", float("inf")) < self.min_cost:
                    self.min_cost = info["cost"]

            cost = info.get('cost', 0)
            self.total_cost += cost

            cost = info.get('cost', 0)
            self.total_cost += cost

            self.episode_metrics.append({
                "episode": self.total_episodes,
                "termination_reason": info.get("termination_reason"),
                "is_success": info.get("termination_reason") == "success",
                "is_unsafe": info.get("is_unsafe", False),
                "cost": cost,
                "reward": reward,
                "episode_length": self.num_timesteps - sum(self.episode_lengths),
                "info": info
            })

        return True

    def get_summary(self, total_timesteps):
        total_eps = self.total_episodes
        total_success = self.success_count
        total_unsafe = self.unsafe_counts
        train_time = time.time() - self.training_start_time

        episode_lengths = self.episode_lengths or [0]
        avg_episode_length = sum(episode_lengths) / total_eps if total_eps > 0 else 0
        episode_length_ci = (0, 0)
        if total_eps >= 2:
            episode_length_ci = stats.t.interval(
                0.95, len(episode_lengths) - 1,
                loc=np.mean(episode_lengths),
                scale=stats.sem(episode_lengths)
            )
        else:
            episode_length_ci = (0, 0)

        avg_cost = self.total_cost / total_success if total_success > 0 else float("inf")
        success_rate = total_success / total_eps if total_eps > 0 else 0
        unsafe_recall = total_unsafe / total_eps if total_eps > 0 else 0
        sample_eff = total_success / total_timesteps if total_timesteps > 0 else 0

        termination_types = Counter(
            ep["termination_reason"] for ep in self.episode_metrics if "termination_reason" in ep
        )

        re1_vals = [ep["info"].get("Re1", 0) for ep in self.episode_metrics if "info" in ep]
        re2_vals = [ep["info"].get("Re2", 0) for ep in self.episode_metrics if "info" in ep]
        re3_vals = [ep["info"].get("Re3", 0) for ep in self.episode_metrics if "info" in ep]

        return {
            "episodes": total_eps,
            "success_count": total_success,
            "success_rate": round(success_rate, 4),
            "unsafe_recall": round(unsafe_recall, 4),
            "avg_cost": round(avg_cost, 2),
            "min_cost": round(self.min_cost, 2) if total_success > 0 else float("inf"),
            "avg_episode_length": round(avg_episode_length, 2),
            "episode_length_ci": tuple(round(x, 2) for x in episode_length_ci),
            "sample_efficiency": round(sample_eff, 4),
            "train_time": round(train_time, 2),
            "total_timesteps": total_timesteps,
            "total_unsafe_episodes": total_unsafe,
            "termination_breakdown": dict(termination_types),
            "avg_re1": round(np.mean(re1_vals), 2) if re1_vals else 0,
            "avg_re2": round(np.mean(re2_vals), 2) if re2_vals else 0,
            "avg_re3": round(np.mean(re3_vals), 2) if re3_vals else 0
        }

class TensorboardCostRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_costs = []

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        reward = self.locals['rewards'][0]
        done = self.locals['dones'][0]

        if done:
            cost = info.get("cost", 0)
            self.episode_rewards.append(reward)
            self.episode_costs.append(cost)

            self.logger.record("episode/mean_reward", reward)
            self.logger.record("episode/mean_cost", cost)

        return True

class ScenicRLTrainer:
    def __init__(self, scenepath, algo, seed, total_timesteps, max_steps,tm_port, log_root,
                 use_ellipse=True, reward_terms=["re1", "re2", "re3"], attack_type="None", ellipse_json_path="ellipse_params.json"):
        self.scenepath = scenepath
        self.seed = seed
        self.total_timesteps = total_timesteps
        self.tm_port = tm_port
        self.max_steps = max_steps
        self.use_ellipse = use_ellipse
        self.reward_terms = reward_terms
        self.attack_type = attack_type

        scene_name = os.path.basename(scenepath).replace(".scenic", "")
        ablation_suffix = f"_ellipse{use_ellipse}_terms{'-'.join(reward_terms)}_attack{attack_type}"
        self.run_name = f"{scene_name}_{algo}_seed{seed}{ablation_suffix}"
        self.log_dir = os.path.join(log_root, self.run_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join("checkpoints", self.run_name)
        os.makedirs("checkpoints", exist_ok=True)

        self.env = ScenicCarlaEnv(
            scenefile=scenepath,
            ellipse_json_path=ellipse_json_path,
            attack_type=attack_type,
            max_steps=max_steps,
            tm_port=tm_port,
            replay_dir=self.log_dir,
            use_ellipse=use_ellipse,
            reward_terms=reward_terms
        )

        algo_map = {
            "PPO": PPO,
            "DQN": DQN,
            "A2C": A2C,
        }
        assert algo in algo_map, f"Unsupported algo: {algo}"
        self.model = algo_map[algo](
            "MlpPolicy", self.env, verbose=1, seed=self.seed, tensorboard_log=self.log_dir)

    def train(self):
        carla_callback = CarlaScenicCallback()
        tb_callback = TensorboardCostRewardCallback()

        callback = CallbackList([
            carla_callback,
            tb_callback
        ])

        self.model.learn(total_timesteps=self.total_timesteps, callback=callback)

        model_path = os.path.join(self.checkpoint_dir, f"{self.run_name}.zip")
        self.model.save(model_path)

        metrics = carla_callback.get_summary(self.total_timesteps)
        metrics["seed"] = self.seed

        json_summary_path = os.path.join(self.log_dir, "summary.json")
        with open(json_summary_path, "w") as f:
            json.dump(metrics, f, indent=2)
        Logger.info(f"[Summary Saved to] {json_summary_path}")

        Logger.info(f"[Result] {self.run_name} | {metrics}")
        return metrics

def run_single_seed(scenepath, algo, seed, total_timesteps, tm_port, max_steps=200, log_root="tensorboard_logs", use_ellipse=True,
                    reward_terms=["re1", "re2", "re3"], attack_type="None", ellipse_json_path="ellipse_params.json"):

    Logger.info(f"Running {algo} on {scenepath} [seed={seed}]")

    try:
        from scenic.syntax.veneer import currentSimulation, endSimulation
        if currentSimulation:
            endSimulation(currentSimulation)
    except Exception as pre_cleanup_error:
        Logger.warning(f"[Seed {seed}] Scenic simulation cleanup failed: {pre_cleanup_error}")

    trainer = None
    result = None

    try:
        trainer = ScenicRLTrainer(
            scenepath=scenepath,
            algo=algo,
            seed=seed,
            total_timesteps=total_timesteps,
            max_steps=max_steps,
            tm_port=tm_port,
            log_root=log_root,
            use_ellipse=use_ellipse,
            reward_terms=reward_terms,
            attack_type=attack_type,
            ellipse_json_path=ellipse_json_path
        )
        result = trainer.train()
        result["seed"] = seed

        summary_path = os.path.join(log_root, f"summary_{algo}_seed{seed}.csv")
        pd.DataFrame([result]).to_csv(summary_path, index=False)

        return result

    except Exception as e:
        raise e
    finally:
        try:
            if currentSimulation:
                endSimulation(currentSimulation)
        except Exception as cleanup_error:
            Logger.warning(f"[Seed {seed}] Scenic simulation failed: {cleanup_error}")

        import gc
        del trainer
        gc.collect()

def run_multiple_seeds(scenepath, algo, start_seed, end_seed, total_timesteps, tm_port,
                      max_steps=500, log_root="tensorboard_logs", **ablation_kwargs):
    all_metrics = []
    for seed in range(start_seed, end_seed + 1):
        Logger.info(f"\n===== Running Seed {seed}/{end_seed} =====")
        try:
            metrics = run_single_seed(
                scenepath=scenepath,
                algo=algo,
                seed=seed,
                total_timesteps=total_timesteps,
                tm_port=tm_port + seed,
                max_steps=max_steps,
                log_root=log_root,** ablation_kwargs
            )
            all_metrics.append(metrics)
        except Exception as e:
            Logger.error(f"Seed {seed} failed: {e}")
            continue

    summary = {
        "algo": algo,
        "seeds": f"{start_seed}-{end_seed}",
        "total_seeds": len(all_metrics),
        **ablation_kwargs
    }

    for key in ["success_rate", "unsafe_recall", "avg_cost", "sample_efficiency", "train_time"]:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            summary[f"{key}_mean"] = round(np.mean(values), 4)
            summary[f"{key}_std"] = round(np.std(values), 4)
            ci = stats.t.interval(0.95, len(values)-1, loc=np.mean(values), scale=stats.sem(values))
            summary[f"{key}_ci95"] = (round(ci[0], 4), round(ci[1], 4))

    summary_path = os.path.join(log_root, f"summary_{algo}_seeds{start_seed}-{end_seed}.csv")
    pd.DataFrame(all_metrics).to_csv(summary_path.replace(".csv", "_raw.csv"), index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    Logger.info(f"[Multi-seed Summary] {summary}")
    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default="scenarios/carlaChallenge10.scenic")
    parser.add_argument('--algo', type=str, choices=['PPO', 'DQN', 'A2C'], default='PPO')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timesteps', type=int, default=50000)
    parser.add_argument('--myport', type=int, default=28622)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--log_root', type=str, default='tensorboard_logs')

    parser.add_argument('--use_ellipse', action='store_false')
    parser.add_argument('--reward_terms', nargs='+', default=["re1", "re2", "re3"])
    parser.add_argument('--attack_type', type=str, default="None",
                        choices=["None", "Spoofing", "TimingDelay", "Jitter"])
    parser.add_argument('--ellipse_json', type=str,
                        default="ellipse_params.json")

    parser.add_argument('--start_seed', type=int, default=None)
    parser.add_argument('--end_seed', type=int, default=None)

    args = parser.parse_args()

    ablation_kwargs = {
        "use_ellipse": args.use_ellipse,
        "reward_terms": args.reward_terms,
        "attack_type": args.attack_type,
        "ellipse_json_path": args.ellipse_json
    }

    if args.start_seed is not None and args.end_seed is not None:
        run_multiple_seeds(
            scenepath=args.scene,
            algo=args.algo,
            start_seed=args.start_seed,
            end_seed=args.end_seed,
            total_timesteps=args.timesteps,
            tm_port=args.myport,
            max_steps=args.max_steps,
            log_root=args.log_root, **ablation_kwargs
        )
    else:
        result = run_single_seed(
            scenepath=args.scene,
            algo=args.algo,
            seed=args.seed,
            total_timesteps=args.timesteps,
            tm_port=args.myport,
            max_steps=args.max_steps,
            log_root=args.log_root,
            **ablation_kwargs
        )
        summary_path = os.path.join(args.log_root, f"summary_{args.algo}_seed{args.seed}.csv")
        pd.DataFrame([result]).to_csv(summary_path, index=False)


