import gymnasium as gym
import numpy as np
import torch
import time
from gymnasium import spaces
from stable_baselines3 import DQN, A2C, DDPG, PPO, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, ActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from adas_env.envs import ADASEnv
from tools.logs import Logger

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.custom_data_list = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.current_action_sequence = []
        self.min_cost = 1e9
        self.avg_cost = 0.0
        self.successful_rate = 0.0
        self.num_success = 0
        self.first_success_episode = None
        self.num_episodes = 0
        self.success_episodes = []
        self.episode_recalls = []
        self.current_recall = 0.0

    def _on_step(self) -> bool:
        done = self.locals['dones'][0]
        reward = self.locals['rewards'][0]
        info = self.locals['infos'][0]
        action = self.locals['actions'][0]

        self.current_episode_reward += reward
        self.current_episode_length += 1
        self.current_action_sequence.append(action)

        if done:
            self.current_recall = info.get('recall', 0.0)
            self.episode_recalls.append(self.current_recall)

            if info['status'] == "success":
                if self.first_success_episode is None:
                    self.first_success_episode = self.num_episodes + 1
                self.min_cost = min(self.min_cost, info['total_cost'])
                self.avg_cost = self.avg_cost * self.num_success + info['total_cost']
                self.num_success += 1
                self.avg_cost /= self.num_success

                self.success_episodes.append({
                    "ep": self.num_episodes + 1,
                    "cost": info['total_cost'],
                    "reward": self.current_episode_reward,
                    "actions": self.current_action_sequence.copy()
                })


            self.num_episodes += 1
            self.successful_rate = self.num_success / self.num_episodes

            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)

            if self.min_cost >= 1e9:
                Logger.info(
                    f'num episodes {self.num_episodes}, average cost {self.avg_cost:.3f}, minimum cost INF, successful rate {self.successful_rate * 100:.3f}%, total cost {info["total_cost"]:.3f}')
            else:
                Logger.info(
                    f'num episodes {self.num_episodes}, average cost {self.avg_cost:.3f}, minimum cost {self.min_cost:.3f}, successful rate {self.successful_rate * 100:.3f}%, total cost {info["total_cost"]:.3f}')
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.current_action_sequence = []

        return True

    def get_avg_recall(self):
        if len(self.episode_recalls) > 0:
            return sum(self.episode_recalls) / len(self.episode_recalls)
        return 0.0

    def get_success_samples(self, indices=[1, 10, 50, 100]):
        result = {}
        for idx in indices:
            if len(self.success_episodes) >= idx:
                result[f"success_ep_{idx}"] = self.success_episodes[idx - 1]
        return result


class RescaledActionEnv(gym.Wrapper):
    def __init__(self, env, low=0, high=1):
        super(RescaledActionEnv, self).__init__(env)
        self.low = low
        self.high = high
        self.action_space = spaces.Box(low=low, high=high, shape=(), dtype=np.float32)

    def rescale_action(self, action):
        rescaled_action = 2 * (action - self.low) / (self.high - self.low) - 1
        return rescaled_action

    def step(self, action):
        rescaled_action = self.rescale_action(action)
        obs, reward, terminated, truncated, info = self.env.step(rescaled_action)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class DRLTrainer:
    def __init__(self, algo='DQN', env_name='ADAS', env_config='S1-dr', learning_rate=1e-4, reward_config=None,  tensorboard_log=None):
        super(DRLTrainer, self).__init__()

        if algo == 'DQN':
            self.env = gym.make(env_name, env_config=env_config, reward_config=reward_config)
            self.model = DQN("MlpPolicy", self.env, verbose=0, tensorboard_log=tensorboard_log, exploration_final_eps=0.01, learning_rate=learning_rate)
        elif algo == 'A2C':
            self.vec_env = make_vec_env(lambda: gym.make(env_name, env_config=env_config, reward_config=reward_config), n_envs=8)
            self.model = A2C("MlpPolicy", self.vec_env, verbose=0, tensorboard_log=tensorboard_log, n_steps=384, learning_rate=learning_rate)
        elif algo == 'PPO':
            self.vec_env = make_vec_env(lambda: gym.make(env_name, env_config=env_config, reward_config=reward_config), n_envs=8)
            self.env = gym.make(env_name, env_config=env_config, reward_config=reward_config)
            self.model = PPO("MlpPolicy", self.vec_env, verbose=0, tensorboard_log=tensorboard_log, n_steps=384, learning_rate=learning_rate, gamma=0.999)
        elif algo == 'DDPG':
            raw_env = gym.make(env_name, env_config=env_config, reward_config=reward_config)
            low = raw_env.unwrapped.action_low
            high = raw_env.unwrapped.action_high
            self.env = RescaledActionEnv(raw_env, low=low, high=high)
            n_actions = 1
            action_noise = NormalActionNoise(mean=np.zeros(n_actions, ), sigma=1.0 * np.ones(n_actions, ))
            self.model = DDPG("MlpPolicy", self.env, verbose=0, tensorboard_log=tensorboard_log, action_noise=action_noise, learning_rate=learning_rate)
        else:
            raise Exception

    def train(self, total_timestamps=100000, log_interval=4, callback=None):
        start = time.time()
        if callback:
            self.model.learn(total_timestamps, log_interval=log_interval, callback=callback)
        else:
            self.model.learn(total_timestamps, log_interval=log_interval)

        end = time.time()
        Logger.info(f'time cost {end - start:.3f}s')

    def save(self, path: str) -> None:
        self.model.save(path)

    def warm_up(self, num_iters):
        model = self.model.policy

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = torch.nn.MSELoss()

        state, info = self.env.reset(seed=7)
        batch_source, batch_target = [], []
        batch_size = 64
        for _ in range(num_iters):
            action = self.env.action_space.n - 1
            observation, reward, terminated, truncated, info = self.env.step(action)
            print(f'in warm up, action {action}')

            if len(batch_source) < batch_size:
                batch_source.append(state.tolist())
                batch_target.append(action)
            else:
                input_data = torch.tensor(batch_source, dtype=torch.float32)

                features = model.extract_features(input_data)
                if model.share_features_extractor:
                    latent_pi, latent_vf = model.mlp_extractor(features)
                else:
                    pi_features, vf_features = features
                    latent_pi = model.mlp_extractor.forward_actor(pi_features)
                    latent_vf = model.mlp_extractor.forward_critic(vf_features)
                values = model.value_net(latent_vf)
                mean_actions = model.action_net(latent_pi)

                distribution = model.action_dist.proba_distribution(action_logits=mean_actions)

                action_dim = self.env.action_space.n
                output_target = torch.pow(torch.arange(1, action_dim + 1), 3.0)
                output_target /= torch.sum(output_target)

                output_target = output_target.unsqueeze(0).repeat_interleave(repeats=batch_size, dim=0)

                actions, _, _ = self.model.policy(input_data)

                optimizer.zero_grad()
                loss = criterion(mean_actions, output_target)
                loss.backward()
                optimizer.step()

                batch_source = [state.tolist()]
                batch_target = [action]

            state = observation

            if terminated or truncated:
                state, info = self.env.reset(seed=7)


if __name__ == '__main__':
    callback = CustomCallback(verbose=1)
    trainer = DRLTrainer(algo='PPO', env_name='ADAS', env_config='In1-dr')

    print(trainer.model.policy)


    trainer.train(total_timestamps=500000, callback=callback)
