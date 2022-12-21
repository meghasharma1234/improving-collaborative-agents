import os

import gym
import numpy as np
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pantheonrl.common.multiagentenv import SimultaneousEnv, PlayerException
from stable_baselines3.ppo import PPO
import torch
from .OvercookedPlay import OvercookedSelfPlayEnv


class OvercookedSelfPlayModifiedEnv(gym.Env):
    def __init__(self, layout_name, embedding_dim=None, alpha=0.5,  baselines=False, initial_teammate_idx=None, rew_mode='multiply'):
        """
        base_env: OvercookedEnv
        featurize_fn: what function is used to featurize states returned in the 'both_agent_obs' field
        """
        super(OvercookedSelfPlayModifiedEnv, self).__init__()

        self.embedding_dim = embedding_dim  # agent embedding dim. from RNN
        self.alpha = alpha   # discriminator reward coefficient
        self.rew_mode = rew_mode

        self._players: Tuple[int, ...] = tuple()
        self._obs: Tuple[Optional[np.ndarray], ...] = tuple()
        self._old_ego_obs: Optional[np.ndarray] = None

        self.total_rews = [0] * 1
        self.ego_moved = False

        DEFAULT_ENV_PARAMS = {
            "horizon": 800,
        }
        rew_shaping_params = {
            "PLACEMENT_IN_POT_REW": 3,
            "DISH_PICKUP_REWARD": 3,
            "SOUP_PICKUP_REWARD": 5,
            "DISH_DISP_DISTANCE_REW": 0,
            "POT_DISTANCE_REW": 0,
            "SOUP_DISTANCE_REW": 0,
        }

        self.mdp = OvercookedGridworld.from_layout_name(layout_name=layout_name, rew_shaping_params=rew_shaping_params)
        mlp = MediumLevelPlanner.from_pickle_or_compute(self.mdp, NO_COUNTERS_PARAMS, force_compute=False)

        self.base_env = OvercookedEnv(self.mdp, **DEFAULT_ENV_PARAMS)
        self.featurize_fn = lambda x: self.mdp.featurize_state(x, mlp)

        if baselines: np.random.seed(0)

        self.observation_space = self._setup_observation_space()
        self.lA = len(Action.ALL_ACTIONS)
        self.action_space  = gym.spaces.Discrete( self.lA )
        self.current_turn = 0

        # potential teammates are in the data folder.
        self.sp_data_dir = os.path.join('data', 'self_play_training_models')
        self.all_teammates = []
        # iterate through all subfolders and load all files
        for root, dirs, files in os.walk(self.sp_data_dir):
            for file in files:
                if file.endswith('.zip'):
                    agent = PPO.load(os.path.join(root, file))
                    self.all_teammates.append(agent)
        self.initial_teammate_idx = initial_teammate_idx

    def setDiscriminatorRewardFn(self, discriminator_reward_fn):
        """Set Disrcriminator Reward Function
        """
        self.discriminator_reward_fn = discriminator_reward_fn

    def setEmbedding(self, embedding_vec):
        """Set embedding vector
        """
        self.embedding_vec = embedding_vec

    def setPolicy(self, learner_policy):
        """
        Set the policy of the learner agent
        Need this so we can extract probabilities later
        Not the best way to do this, but it works
        """
        self.learner_policy = learner_policy

    def calcDiscriminatorReward(self, obs, acts, next_state, done, p_a):
        """Calculate the reward from discriminator times constant
        """
        discriminator_out = self.discriminator_reward_fn(obs, acts, next_state, done, p_a)
        return self.alpha * discriminator_out

    def _setup_observation_space(self):
        # dummy_state = self.mdp.get_standard_start_state()
        # obs_shape = self.featurize_fn(dummy_state)[0].shape
        # high = np.ones(obs_shape, dtype=np.float32) * np.inf  # max(self.mdp.soup_cooking_time, self.mdp.num_items_for_soup, 5)

        # modify for embedding
        embedding_dim = self.embedding_dim
        original_dim = 62
        new_dim = embedding_dim + original_dim if self.embedding_dim is not None else original_dim

        high = np.ones((new_dim,), dtype=np.float32) * np.inf
        return gym.spaces.Box(-high, high, dtype=np.float64)

    def getDummyEnv(self, player_num: int):
        """
        Returns a dummy environment with just an observation and action
        space that a partner agent can use to construct their policy network.

        :param player_num: the partner number to query
        """
        return self

    def get_teammate_action(self, obs):
        """Get teammate action
        """
        teammate_action, _states = self.teammate.predict(obs)
        return teammate_action

    def _get_actions(self, current_player_action):
        # by default, stay still
        actions = [4, 4]
        teammate_action = self.get_teammate_action(self.old_other_obs)
        actions[self.ego_idx] = current_player_action
        actions[(self.ego_idx + 1) % 2] = teammate_action
        return np.array(actions)

    def multi_step(self, ego_action, alt_action):
        """
        action:
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in index format
            encoded as an int

        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """
        ego_action, alt_action = Action.INDEX_TO_ACTION[ego_action], Action.INDEX_TO_ACTION[alt_action]
        joint_action = (ego_action, alt_action)

        next_state, reward, done, info = self.base_env.step(joint_action)

        # reward shaping
        rew_shape = info['shaped_r']
        reward = reward + rew_shape

        #print(self.base_env.mdp.state_string(next_state))
        ob_p0, ob_p1 = self.featurize_fn(next_state)
        ego_obs, alt_obs = ob_p0, ob_p1

        return (ego_obs, alt_obs), (reward, reward), done, {}#info

    def n_step(
                    self,
                    actions: List[np.ndarray],
                ) -> Tuple[Tuple[int, ...],
                           Tuple[Optional[np.ndarray], ...],
                           Tuple[float, ...],
                           bool,
                           Dict]:
        return ((0, 1),) + self.multi_step(actions[0], actions[1])

    def step(
                self,
                action: np.ndarray
            ) -> Tuple[Optional[np.ndarray], float, bool, Dict]:
        """
        Run one timestep from the perspective of the ego-agent. This involves
        calling the ego_step function and the alt_step function to get to the
        next observation of the ego agent.

        Accepts the ego-agent's action and returns a tuple of (observation,
        reward, done, info) from the perspective of the ego agent.

        :param action: An action provided by the ego-agent.

        :returns:
            observation: Ego-agent's next observation
            reward: Amount of reward returned after previous action
            done: Whether the episode has ended (need to call reset() if True)
            info: Extra information about the environment
        """
        acts = self._get_actions(action)
        self._players, self._obs, rews, done, info = self.n_step(acts)
        rews = (float(rews[0]), float(rews[1]))

        rew = rews[self.ego_idx]

        if done:
            return self._old_ego_obs, rew, done, info

        current_player = self.current_turn
        
        ego_obs = self._obs[self.ego_idx]
        self.old_other_obs = self._obs[(self.ego_idx + 1) % 2]
        # calculate new reward
        # if self.rew_mode == 'multiply':
        #     rew *= self.calcDiscriminatorReward(self._old_ego_obs, acts, ego_obs, done) / self.alpha
        # else:
        #     rew += self.calcDiscriminatorReward(self._old_ego_obs, acts, ego_obs, done)
        batch_obs = torch.Tensor(self._old_ego_obs).unsqueeze(0).cuda()
        p_a = self.learner_policy.policy.get_distribution(batch_obs). \
                        distribution.probs.squeeze(0).cpu().detach().numpy()[action]
        # p_a = 1 / 6
        rew *= self.calcDiscriminatorReward(self._old_ego_obs, acts, ego_obs, done, p_a)
        self._old_ego_obs = ego_obs
        
        # append embedding to make new observation
        if self.embedding_dim is not None:
            ego_obs = np.concatenate((self.embedding_vec, ego_obs))
        else:
            ego_obs = ego_obs
        return ego_obs, rew, done, info

    def reset(self) -> np.ndarray:
        """
        Reset environment to an initial state and return the first observation
        for the ego agent.

        :returns: Ego-agent's first observation
        """
        self._players, self._obs = self.n_reset()
        # generate random idx either 0 or 1
        self.ego_idx = np.random.randint(2)
        self.old_other_obs = self._obs[(self.ego_idx + 1) % 2]
        self.total_rews = [0] * 1
        self.ego_moved = False
        if self.initial_teammate_idx is None:
            self.teammate_idx = np.random.randint(len(self.all_teammates))
        else:
            self.teammate_idx = self.initial_teammate_idx
        self.teammate = self.all_teammates[self.teammate_idx]

        while 0 not in self._players:
            acts = self._get_actions()
            self._players, self._obs, rews, done, _ = self.n_step(acts)

            if done:
                raise PlayerException("Game ended before ego moved")


        ego_obs = self._obs[self.ego_idx]
        self.old_other_obs = self._obs[(self.ego_idx + 1) % 2]

        assert ego_obs is not None
        self._old_ego_obs = ego_obs

        # append embedding to make new observation
        if self.embedding_dim is not None:
            ego_obs = np.concatenate((self.embedding_vec, ego_obs))
        else:
            ego_obs = ego_obs
        return ego_obs

    def n_reset(self) -> Tuple[Tuple[int, ...],
                               Tuple[Optional[np.ndarray], ...]]:
        return (0, 1), self.multi_reset()

    def multi_reset(self) -> np.ndarray:
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """
        self.base_env.reset()
        ob_p0, ob_p1 = self.featurize_fn(self.base_env.state)
        ego_obs, alt_obs = ob_p0, ob_p1

        return (ego_obs, alt_obs)

    def render(self, mode='human', close=False):
        pass
