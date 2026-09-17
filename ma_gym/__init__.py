import logging

from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='FindTreasure-v0',
    entry_point='ma_gym.envs.find_treasure:FindTreasure',
)

register(
    id='PGM-3ag-v0',
    entry_point='ma_gym.envs.patient_gold_miner:PGM',
)

register(
    id='PGM-6ag-v0',
    entry_point='ma_gym.envs.patient_gold_miner:PGM',
    kwargs={'grid_shape': (12, 12), 'n_agents': 6, 'n_golds': 2, 'n_stones': 3,
            'gold_punish_steps': 10, 'stone_reward': 0.3, 'step_cost': -0.2,
            'gold_base_punish': -1, 'gold_reward': 30, 'max_steps': 50,
            'stone_disappear_steps': 10, 'agent_view': (2, 2)}
)
