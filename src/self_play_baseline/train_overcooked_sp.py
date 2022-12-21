"""
This is a simple example training script for PantheonRL.

To run this script, remember to first install overcooked
via the instructions in the README.md
"""
import os

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from pantheonrl.common.agents import OnPolicyAgent
from overcookedgym.overcooked_utils import LAYOUT_LIST
from src.gym_overcooked.envs.OvercookedPlay import OvercookedSelfPlayEnv
import tqdm
import numpy as np
import shutil
from stable_baselines3.common.results_plotter import load_results, ts2xy

class CheckpointCallbackWithRew(CheckpointCallback):
    def __init__(self, n_steps, save_freq, save_path, name_prefix, save_replay_buffer, save_vecnormalize,
                 initial_model_path, medium_model_path, final_model_path, verbose=0):
        super().__init__(save_freq, save_path, name_prefix, save_replay_buffer, save_vecnormalize, verbose)
        self.initial_model_path = initial_model_path
        self.medium_model_path = medium_model_path
        self.final_model_path = final_model_path
        self.n_steps = n_steps
        self.best_mean_reward = -np.inf
        self.all_rewards = []
        self.all_save_paths = []

    def _on_step(self) -> bool:
        super()._on_step()
        if self.n_calls % self.save_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.save_path), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
                model_path = self._checkpoint_path(extension="zip")
                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {model_path} with mean reward {mean_reward:.2f}")
                    self.model.save(self.final_model_path)
                self.all_rewards.append(mean_reward)
                self.all_save_paths.append(model_path)
            if self.n_calls == self.n_steps:
                # save initial model
                shutil.copy(self.all_save_paths[0], self.initial_model_path)

                # save second best model
                def find_closest_idx(arr, val):
                    idx = np.abs(arr - val).argmin()
                    val = arr[idx]
                    return idx, val

                second_best_reward_idx, reward = find_closest_idx(np.array(self.all_rewards), self.best_mean_reward * 0.5)
                if self.verbose > 0:
                    print(f"Saving medium model to {self.medium_model_path} with mean reward {reward:.2f}")
                shutil.copy(self.all_save_paths[second_best_reward_idx], self.medium_model_path)
        return True

def trainSelfPlay():
    # Since pantheonrl's MultiAgentEnv is a subclass of the gym Env, you can
    # register an environment and construct it using gym.make.
    # env = OvercookedSelfPlayEnv(layout_name=layout)
    # env = gym.make("gym_overcooked/OvercookedSelfPlayEnv-v0", layout_name='simple')
    n_agents = 30
    N_steps = 500000
    checkpoint_freq = N_steps // 100
    layout = 'simple'
    assert layout in LAYOUT_LIST

    for i in tqdm.tqdm(range(n_agents)):

        # env = OvercookedSelfPlayEnv(layout_name='simple', baselines=True)
        # env = gym.make("gym_overcooked/OvercookedSelfPlayEnv-v0", layout_name='simple', baselines=True)
        env = OvercookedSelfPlayEnv(layout_name=layout, baselines=True)

        # gym_env = get_vectorized_gym_env(
        #   env, 'Overcooked-v0', featurize_fn=lambda x: mdp.lossless_state_encoding(x), **params
        # )

        # Before training your ego agent, you first need to add your partner agents
        # to the environment. You can create adaptive partner agents using
        # OnPolicyAgent (for PPO/A2C) or OffPolicyAgent (for DQN/SAC). If you set
        # verbose to true for these agents, you can also see their learning progress


        seed = i
        # Finally, you can construct an ego agent and train it in the environment
        # ego = PPO('MlpPolicy', env, verbose=1)

        initial_model_path = os.path.join('data', 'self_play_training_models', 'seed_' + str(seed), 'initial_model.zip')
        medium_model_path = os.path.join('data',  'self_play_training_models', 'seed_' + str(seed), 'medium_model.zip')
        final_model_path = os.path.join('data',  'self_play_training_models', 'seed_' + str(seed), 'final_model.zip')

        save_dir = os.path.join('data', 'ppo_self_play', 'seed_{}'.format(seed))
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        checkpoint_callback = CheckpointCallbackWithRew(
          n_steps = N_steps,
          save_freq=checkpoint_freq,
          save_path=save_dir,
          name_prefix="rl_model",
          save_replay_buffer=True,
          save_vecnormalize=True,
          initial_model_path=initial_model_path,
          medium_model_path=medium_model_path,
          final_model_path=final_model_path,
        )

        env = Monitor(env, "./" + save_dir + "/")
        agent = PPO('MlpPolicy', env, verbose=0, seed=seed)
        agent.learn(total_timesteps=N_steps, callback=checkpoint_callback)

        # To visualize the agent:
        # python overcookedgym/overcooked-flask/app.py --modelpath_p0 ../logs/rl_model_500000_steps --modelpath_p1 ../logs/rl_model_50000_steps --layout_name simple

if __name__ == '__main__':
    trainSelfPlay()