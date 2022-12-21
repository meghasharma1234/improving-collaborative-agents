from gym.envs.registration import register

register(
    id='gym_overcooked/OvercookedSelfPlayEnv-v0',
    entry_point='gym_overcooked.envs:OvercookedSelfPlayEnv',
    max_episode_steps=800,
    kwargs={'layout_name': 'simple'}
)

register(
    id='gym_overcooked/OvercookedSelfPlayModifiedEnv-v0',
    entry_point='gym_overcooked.envs:OvercookedSelfPlayModifiedEnv',
    max_episode_steps=800,
    kwargs={'layout_name': 'simple'}
)