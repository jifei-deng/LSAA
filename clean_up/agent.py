

from gym.spaces import Box
from gym.spaces import Discrete
import numpy as np
import clean_up.utils.utility_funcs as util


BASE_ACTIONS = {0: 'MOVE_LEFT',
                1: 'MOVE_RIGHT',
                2: 'MOVE_UP',
                3: 'MOVE_DOWN',
                4: 'STAY',
                5: 'TURN_CLOCKWISE',
                6: 'TURN_COUNTERCLOCKWISE'}


class Agent(object):

    def __init__(self, agent_id, start_pos, start_orientation, grid, row_size, col_size,
                 global_ref_point=None):

        self.agent_id = agent_id
        self.pos = np.array(start_pos)
        self.orientation = start_orientation

        self.grid = grid
        self.row_size = row_size
        self.col_size = col_size
        self.global_ref_point = global_ref_point
        self.reward_this_turn = 0

    @property
    def action_space(self):

        raise NotImplementedError

    @property
    def observation_space(self):

        raise NotImplementedError

    def action_map(self, action_number):

        raise NotImplementedError

    def get_state(self):
        pos = self.global_ref_point if self.global_ref_point else self.get_pos()
        return util.return_view(self.grid, pos,
                                self.row_size, self.col_size)

    def compute_reward(self):
        reward = self.reward_this_turn
        self.reward_this_turn = 0
        return reward

    def set_pos(self, new_pos):
        self.pos = np.array(new_pos)

    def get_pos(self):
        return self.pos

    def translate_pos_to_egocentric_coord(self, pos):
        offset_pos = pos - self.get_pos()
        ego_centre = [self.row_size, self.col_size]
        return ego_centre + offset_pos

    def set_orientation(self, new_orientation):
        self.orientation = new_orientation

    def get_orientation(self):
        return self.orientation

    def get_map(self):
        return self.grid

    def return_valid_pos(self, new_pos):

        ego_new_pos = new_pos
        new_row, new_col = ego_new_pos

        temp_pos = new_pos.copy()
        if self.grid[new_row, new_col] == '@':
            temp_pos = self.get_pos()
        return temp_pos

    def update_agent_pos(self, new_pos):

        old_pos = self.get_pos()
        ego_new_pos = new_pos
        new_row, new_col = ego_new_pos

        temp_pos = new_pos.copy()
        if self.grid[new_row, new_col] == '@':
            temp_pos = self.get_pos()
        self.set_pos(temp_pos)

        return self.get_pos(), np.array(old_pos)

    def update_agent_rot(self, new_rot):
        self.set_orientation(new_rot)

    def hit(self, char):

        raise NotImplementedError

    def consume(self, char):

        raise NotImplementedError


HARVEST_ACTIONS = BASE_ACTIONS.copy()
HARVEST_ACTIONS.update({7: 'FIRE'})

HARVEST_VIEW_SIZE = 7


class HarvestAgent(Agent):

    def __init__(self, agent_id, start_pos, start_orientation, grid, view_len=HARVEST_VIEW_SIZE):
        self.view_len = view_len
        super().__init__(agent_id, start_pos, start_orientation, grid, view_len, view_len)
        self.update_agent_pos(start_pos)
        self.update_agent_rot(start_orientation)

    @property
    def action_space(self):
        return Discrete(8)



    def action_map(self, action_number):

        return HARVEST_ACTIONS[action_number]

    @property
    def observation_space(self):
        return Box(low=0.0, high=0.0, shape=(2 * self.view_len + 1,
                                             2 * self.view_len + 1, 3), dtype=np.float32)

    def hit(self, char):
        if char == 'F':
            self.reward_this_turn -= 50

    def fire_beam(self, char):
        if char == 'F':
            self.reward_this_turn -= 1

    def get_done(self):
        return False

    def consume(self, char):

        if char == 'A':
            self.reward_this_turn += 1
            return ' '
        else:
            return char


CLEANUP_ACTIONS = BASE_ACTIONS.copy()
CLEANUP_ACTIONS.update({7: 'FIRE',
                        8: 'CLEAN'})

CLEANUP_VIEW_SIZE = 7


class CleanupAgent(Agent):
    def __init__(self, agent_id, start_pos, start_orientation, grid, view_len=CLEANUP_VIEW_SIZE,
                 global_ref_point=None, config=None):
        self.config = config
        self.view_len = view_len
        super().__init__(agent_id, start_pos, start_orientation, grid, view_len, view_len,
                         global_ref_point)

        self.update_agent_pos(start_pos)
        self.update_agent_rot(start_orientation)

    @property
    def action_space(self):
        return Discrete(9)

    @property
    def observation_space(self):
        return Box(low=0.0, high=0.0, shape=(2 * self.view_len + 1,
                                             2 * self.view_len + 1, 3), dtype=np.float32)



    def action_map(self, action_number):

        return CLEANUP_ACTIONS[action_number]

    def fire_beam(self, char):
        if char == 'C':
            self.reward_this_turn += self.config.cleaning_penalty

    def get_done(self):
        return False

    def hit(self, char):
        if char == 'F':
            self.reward_this_turn -= 50

    def consume(self, char):

        if char == 'A':
            self.reward_this_turn += self.config.reward_value
            return ' '
        else:
            return char
