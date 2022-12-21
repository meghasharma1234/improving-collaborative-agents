import os
import numpy as np
import gym
import copy 
import torch
from pathlib import Path
import datetime
from collections import Counter

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from imitation.data import rollout
from imitation.scripts.train_adversarial import save
from imitation.algorithms.adversarial.airl import AIRL
# from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.data import types

import shutil
import pandas as pd
from stable_baselines3.common.results_plotter import load_results, ts2xy
from gym.spaces.box import Box
from human_data.generate_human_trajectories import generate_human_trajectories

import gym_overcooked

# class CheckpointCallbackModified(CheckpointCallback):
#     def __init__(self, n_steps, save_freq, save_path, name_prefix, save_replay_buffer, save_vecnormalize,
#                  rew_csv_path, verbose=0):
#         super().__init__(save_freq, save_path, name_prefix, save_replay_buffer, save_vecnormalize, verbose)
#         self.n_steps = n_steps
#         self.best_mean_reward = -np.inf
#         self.df = pd.DataFrame(columns=['timesteps', 'mean_reward'])
#         self.rew_csv_path = rew_csv_path

#     def _on_step(self) -> bool:
#         super()._on_step()
#         if self.n_calls % self.save_freq == 0:
#             # Retrieve training reward
#             x, y = ts2xy(load_results(self.save_path), "timesteps")
#             if len(x) > 0:
#                 # Mean training reward over the last 100 episodes
#                 mean_reward = np.mean(y[-100:])
#                 self.df = self.df.append({'timesteps': self.n_calls, 'mean_reward': mean_reward}, ignore_index=True)
#             if self.n_calls == self.n_steps:
#                 # save csv
#                 self.df.to_csv(self.rew_csv_path, index=False)
#         return True


class AIRLTrainer(object):
    """Main class for training selfPlay agent, generating rollouts, training AIRL, and evaluting policy
    """
    def __init__(self, hparams) -> None:
        self.rng = np.random.default_rng(0)
        self.env_name = "gym_overcooked/OvercookedSelfPlayEnv-v0"
        self.mod_env_name = "gym_overcooked/OvercookedSelfPlayModifiedEnv-v0"
        self.hparams = hparams
        self.rollouts = None
        self.airl_learner = None
        self.loaded_discriminator = None
        self.directory = './logs/' + datetime.datetime.now().strftime("%m%d_%H_%M/")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.policy_kwargs = {}
        h_sz = 256
        self.policy_kwargs = {'net_arch': [dict(pi=[h_sz, h_sz, h_sz, h_sz],
                                                 vf=[h_sz, h_sz, h_sz, h_sz])]
                             }

    def trainExpertPolicy(self):
        """Trains an expert policy

        Returns:
            expert policy (stable_baselines policy)
        """
        print("\nTraining expert policy...")
        env = gym.make(self.env_name, layout_name=self.hparams['layout_name'])
        expert = PPO(policy=MlpPolicy, env=env, policy_kwargs=self.policy_kwargs,
                     tensorboard_log=self.directory + 'expert_selfplay/')
        checkpoint_callback = CheckpointCallback(
            save_freq=self.hparams['trainExpertPolicy_steps'] // 100,
            save_path=self.directory + 'expert_selfplay/',
            name_prefix="rl_model",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
        expert.learn(self.hparams['trainExpertPolicy_steps'], callback=checkpoint_callback, progress_bar=True)
        return expert
    
    def loadExpertPolicy(self):
        """loads agent 

        Args:
            path: path to PPO agent

        Returns:
            expert (stable_baselines policy)
        """
        return PPO.load(path=self.hparams['loadExpertPolicy_path'])

    def loadRollouts(self, human=False, filter_stationary = True):
        """loads rollouts from files starting at 0 filename idx.
        """
        print("\nLoading rollouts...")
        if not human:
            data = []
            idx = 0
            while os.path.exists(f'./data/airl_demonstration/data_rollouts_{idx}.npy'):
                path = f'./data/airl_demonstration/data_rollouts_{idx}.npy'
                data.extend(np.load(path, self.rollouts, allow_pickle=True))
                idx += 1
            self.rollouts = data
        else:
            self.rollouts = generate_human_trajectories()
        
        if filter_stationary and human:
            for i in range(len(self.rollouts)):
                traj = self.rollouts[i]
                idx = np.logical_or(np.random.rand(*traj.acts.shape) < 0.02, (traj.acts != 4))
                trajectory = types.TrajectoryWithRew(
                    obs=traj.obs[np.concatenate((idx, [True]))], # obs are 1 more than acts & last obs has no action
                    acts=traj.acts[idx],
                    rews=traj.rews[idx],
                    infos=[{} for _ in range(len(traj.acts[idx]))],
                    terminal=True
                )
                self.rollouts[i] = trajectory
        print('total trajectories read:', len(self.rollouts))
        print(Counter(self.rollouts[0].acts))
        print(self.rollouts[1].obs.shape, self.rollouts[1].acts.shape)


    def generateRollouts(self, expert, save_file=False):
        """Generates demonstration data rollouts based on given (expert) policy 

        Args:
            expert (stable_baselines policy)
        """
        print("\nGenerating trajectories...")
        self.rollouts = rollout.rollout(
            expert,
            make_vec_env(
                self.env_name, 
                env_make_kwargs={'layout_name': self.hparams['layout_name']},
                rng=self.rng,
                n_envs=5,
                post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
            ),
            rollout.make_sample_until(min_timesteps=None, min_episodes=self.hparams['rollouts_min_episodes']),
            rng=self.rng,
            verbose=True,
        )

        if save_file:
            idx = 0
            while os.path.exists(f'./data/airl_demonstration/data_rollouts_{idx}.npy'):
                idx += 1
            np.save(f'./data/airl_demonstration/data_rollouts_{idx}.npy', self.rollouts, allow_pickle=True)
        print('total trajectories generated:', len(self.rollouts))
    

    def trainAIRL(self):
        """Trains AIRL using rollouts
        """
        print("\nTraining AIRL...")
        assert self.rollouts, "call to generateRollouts OR loadRollouts must be made first"

        venv = make_vec_env(self.env_name, env_make_kwargs={'layout_name': self.hparams['layout_name']},
                            rng=self.rng, n_envs=8)
        
        self.airl_learner = PPO(env=venv, policy=MlpPolicy, policy_kwargs=self.policy_kwargs)
        h_sz = 256
        reward_net = BasicShapedRewardNet(
            venv.observation_space,
            venv.action_space,
            normalize_input_layer=RunningNorm,
            reward_hid_sizes=(h_sz, h_sz),
            potential_hid_sizes=(h_sz, h_sz, h_sz)
        )
        self.airl_trainer = AIRL(
            demonstrations=self.rollouts,
            demo_batch_size=1024,
            gen_replay_buffer_capacity=2048,
            n_disc_updates_per_round=4,
            venv=venv,
            gen_algo=self.airl_learner,
            reward_net=reward_net,
            allow_variable_horizon = True, # added this (gives warning for stability)
            log_dir = self.directory + 'airl',
            init_tensorboard = True,
        )
        def callbackFn(round_num: int, /) -> None:
            if round_num % 5 == 0:
                save(self.airl_trainer, Path(self.directory + 'airl') / f"checkpoint_{round_num:05d}")
        
        self.airl_trainer.train(self.hparams['trainAIRL_steps'], callback=callbackFn)


    def evaluatePolicy(self, policy):
        """Evalutes given policy 

        Args:
            policy (stable_baselines policy): policy to evaluate
        """
        venv = make_vec_env(self.env_name, env_make_kwargs={'layout_name': self.hparams['layout_name']},
                            rng=self.rng, n_envs=8)
        rewards, _ = evaluate_policy(policy, venv, 100, return_episode_rewards=True)
        print("Rewards:", rewards)
    

    def getDiscriminator(self):
        """Returns the discrimanotor network from file
        NOTE: DONE signal must be of shape 1 for entire seq passed to the discirminator
        Args:
            path (str, optional): path to 'reward_train.pt' file.

        Returns:
            BasicShapedRewardNet: The discriminator network 
        """
        self.loaded_discriminator = torch.load(self.hparams['discriminator_loadpath'])

        return self.loaded_discriminator


    def disiciminatorRewardFn(self, obs, act, next_state, done, p_a):
        """Calculates the reward output of the discriminator

        Args:
            obs (numpy array):
            act (numpy array):
            next_state (numpy array):
            done (int (bool)):

        # sample: 
            trajlen = 1
            obs = torch.randn(trajlen, 62).to(airl_trainer.device)
            acts = torch.randn(trajlen, 6).to(airl_trainer.device)  # 6 discrete actions
            next_state = torch.rand_like(obs).to(airl_trainer.device)
            done = torch.randn(1).to(airl_trainer.device)  # MUST be of size 1 used for discount factor calc. 

        Returns:
            float: discriminator reward
        """
        obs = torch.from_numpy(obs).unsqueeze(0).type(torch.float32).to(self.device)
        acts = np.zeros(6)
        acts[act] = 1
        acts = torch.from_numpy(acts).unsqueeze(0).type(torch.float32).to(self.device)  # 6 discrete actions (create one-hot)
        next_state = torch.from_numpy(next_state).unsqueeze(0).type(torch.float32).to(self.device)
        done = torch.tensor(float(done)).unsqueeze(0).type(torch.float32).to(self.device)  # MUST be of size 1 used for discount factor calc.

        reward_out_raw = self.loaded_discriminator(obs, acts, next_state, done)
        reward_out = reward_out_raw.cpu().data.item()
        exp_reward_out = np.exp(reward_out)
        return exp_reward_out / (exp_reward_out + p_a)


    def trainModifiedAgent(self, use_embedding=False):
        """Trains the main agent

        Returns:
            main policy (stable_baselines policy):
        """
        assert self.loaded_discriminator, "Must get disriminator first"
        
        print("\nTraining Modified policy...")

        if use_embedding:
            embedding_size = 32
            env = gym.make(self.mod_env_name, layout_name=self.hparams['layout_name'], embedding_dim=embedding_size, alpha=0.5)
            embedding = np.random.randn(embedding_size)  ## get from RNN classifier
            env.setEmbedding(embedding)
        else:
            env = gym.make(self.mod_env_name, layout_name=self.hparams['layout_name'], embedding_dim=None, alpha=0.5)

        env.setDiscriminatorRewardFn(self.disiciminatorRewardFn)

        expert = PPO(policy=MlpPolicy, env=env, tensorboard_log=self.directory + 'modified_agent/')
        env.setPolicy(expert)
        #checkpoint_callback = CheckpointCallbackModified(
        checkpoint_callback = CheckpointCallback(
            save_freq=self.hparams['trainModExpertPolicy_steps'] // 100,
            save_path=self.directory + 'modified_agent/',
            name_prefix="rl_model",
            save_replay_buffer=True,
            save_vecnormalize=True,
            #rew_csv_path=self.directory + 'modified_agent/modified_agent_rewards.csv'
        )
        expert.learn(self.hparams['trainModExpertPolicy_steps'], callback=checkpoint_callback)

        print("\nEvaluating agent:....")
        rewards, _ = evaluate_policy(expert, env, 100, return_episode_rewards=True)
        print("Rewards:", rewards)
        return expert


