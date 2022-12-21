"""Methods to collect, analyze and manipulate transition and trajectory rollouts."""

from typing import (
    List,
    Sequence,
)

import numpy as np
from overcooked_ai_py.mdp.actions import Action
import pandas as pd
from tqdm import tqdm

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, ObjectState, PlayerState
from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS
from imitation.data import types


def json_state_to_python_state(mdp, df_state):
    """Convert from a df cell format of a state to an Overcooked State"""
    if type(df_state) is str:
        df_state = eval(df_state)

    player_0 = df_state['players'][0]
    player_1 = df_state['players'][1]

    pos_0, or_0 = tuple(player_0['position']), tuple(player_0['orientation'])
    pos_1, or_1 = tuple(player_1['position']), tuple(player_1['orientation'])
    obj_0, obj_1 = None, None

    if 'held_object' in player_0.keys():
        obj_0 = json_obj_to_python_obj_state(mdp, player_0['held_object'])

    if 'held_object' in player_1.keys():
        obj_1 = json_obj_to_python_obj_state(mdp, player_1['held_object'])

    player_state_0 = PlayerState(pos_0, or_0, obj_0)
    player_state_1 = PlayerState(pos_1, or_1, obj_1)

    world_objects = {}
    for obj in df_state['objects'].values():
        object_state = json_obj_to_python_obj_state(mdp, obj)
        world_objects[object_state.position] = object_state

    assert not df_state["pot_explosion"]
    overcooked_state = OvercookedState(players=(player_state_0, player_state_1),
                                       objects=world_objects,
                                       order_list=None)
    return overcooked_state


def json_obj_to_python_obj_state(mdp, df_object):
    """Translates from a df cell format of a state to an Overcooked State"""
    obj_pos = tuple(df_object['position'])
    if 'state' in df_object.keys():
        soup_type, num_items, cook_time = tuple(df_object['state'])

        # Fix differing dynamics from Amazon Turk version
        if cook_time > mdp.soup_cooking_time:
            cook_time = mdp.soup_cooking_time

        obj_state = (soup_type, num_items, cook_time)
    else:
        obj_state = None
    return ObjectState(df_object['name'], obj_pos, obj_state)

class Episode:
    def __init__(self, observations, actions, rewards, dones):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.current_step = 0

    def __len__(self):
        return len(self.observations)

    def get_next(self):
        if self.current_step >= len(self):
            raise IndexError("Episode is exhausted")
        obs = self.observations[self.current_step]
        act = self.actions[self.current_step]
        rew = self.rewards[self.current_step]
        done = self.dones[self.current_step]
        self.current_step += 1
        return obs, act, rew, done

import re
import ast
def str2array(s):
    # Remove space after [
    s=re.sub('\[ +', '[', s.strip())
    # Replace commas and spaces
    s=re.sub('[,\s]+', ', ', s)
    return np.array(ast.literal_eval(s))

def hh_data_to_episodes(data: pd.DataFrame) -> List[Episode]:
    # for now, only get cramped_corridor
    data = data[data["layout_name"] == "cramped_room"]


    episodes = []

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

    mdp = OvercookedGridworld.from_layout_name(layout_name='simple', rew_shaping_params=rew_shaping_params)
    mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=False)

    base_env = OvercookedEnv(mdp, **DEFAULT_ENV_PARAMS)
    featurize_fn = lambda x: mdp.featurize_state(x, mlp)

    current_episode = 0

    curr_ep_obs_p1 = []
    curr_ep_acts_p1 = []
    curr_ep_obs_p2 = []
    curr_ep_acts_p2 = []
    curr_ep_rews = []
    curr_ep_dones = []

    timesteps = data["cur_gameloop"].values
    states = data["state"]
    joint_actions = data["joint_action"]
    rewards = data["reward"]

    print("\nProcessing human rollouts...")

    for i in tqdm(range(len(data) + 1)):
        # finish episode when cur_gameloop stops increasing
        if i == len(data) or (i > 0 and timesteps[i] < timesteps[i - 1]):
            current_episode += 1
            curr_ep_dones[-1] = True
            episodes.append(
                Episode(
                    observations=np.array(curr_ep_obs_p1),
                    actions=np.array(curr_ep_acts_p1[1:]),
                    rewards=np.array(curr_ep_rews[1:]),
                    dones=np.array(curr_ep_dones[1:]),
                )
            )
            episodes.append(
                Episode(
                    observations=np.array(curr_ep_obs_p2),
                    actions=np.array(curr_ep_acts_p2[1:]),
                    rewards=np.array(curr_ep_rews[1:]),
                    dones=np.array(curr_ep_dones[1:]),
                )
            )
            curr_ep_obs_p1 = []
            curr_ep_acts_p1 = []
            curr_ep_obs_p2 = []
            curr_ep_acts_p2 = []
            curr_ep_rews = []
            curr_ep_dones = []

        if i < len(data):
            state_str = states.iloc[i]
            if not isinstance(state_str, str):
                continue
            state = json_state_to_python_state(mdp, state_str)
            obs_p1, obs_p2 = featurize_fn(state)
            action_str = joint_actions.iloc[i]
            if not isinstance(action_str, str):
                continue
            joint_action = str2array(action_str).tolist()
            act_p1, act_p2 = joint_action
            if not isinstance(act_p1, str):
                act_p1 = tuple(act_p1)
            else:
                act_p1 = act_p1.lower()
            if not isinstance(act_p2, str):
                act_p2 = tuple(act_p2)
            else:
                act_p2 = act_p2.lower()
            act_p1, act_p2 = Action.ACTION_TO_INDEX[act_p1], Action.ACTION_TO_INDEX[act_p2]
            rew = rewards.iloc[i]
            done = False

            curr_ep_obs_p1.append(obs_p1)
            curr_ep_acts_p1.append(act_p1)
            curr_ep_obs_p2.append(obs_p2)
            curr_ep_acts_p2.append(act_p2)
            curr_ep_rews.append(rew)
            curr_ep_dones.append(done)
    return episodes



def generate_human_trajectories() -> Sequence[types.TrajectoryWithRew]:
    """Generate trajectory dictionaries from a policy and an environment.
    Args:
        policy: Can be any of the following:
            1) A stable_baselines3 policy or algorithm trained on the gym environment.
            2) A Callable that takes an ndarray of observations and returns an ndarray
            of corresponding actions.
            3) None, in which case actions will be sampled randomly.
        venv: The vectorized environments to interact with.
        sample_until: A function determining the termination condition.
            It takes a sequence of trajectories, and returns a bool.
            Most users will want to use one of `min_episodes` or `min_timesteps`.
        deterministic_policy: If True, asks policy to deterministically return
            action. Note the trajectories might still be non-deterministic if the
            environment has non-determinism!
        rng: used for shuffling trajectories.
    Returns:
        Sequence of trajectories, satisfying `sample_until`. Additional trajectories
        may be collected to avoid biasing process towards short episodes; the user
        should truncate if required.
    """

    df = pd.read_csv("data/trials_hh.csv")
    episodes = hh_data_to_episodes(df)

    trajectories = []
    for episode in episodes:
        trajectory = types.TrajectoryWithRew(
            obs=episode.observations,
            acts=episode.actions,
            rews=episode.rewards,
            infos=[{} for _ in range(len(episode.actions))],
            terminal=True
        )
        trajectories.append(trajectory)
    return trajectories

if __name__ == "__main__":
    trajectories = generate_human_trajectories()


