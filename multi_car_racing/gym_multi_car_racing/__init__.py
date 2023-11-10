from .multi_car_racing import MultiCarRacing

from gymnasium.envs.registration import register

register(
    id='MultiCarRacing-v1',
    entry_point='gym_multi_car_racing:MultiCarRacing',
    max_episode_steps=7000,
    reward_threshold=9000
)
