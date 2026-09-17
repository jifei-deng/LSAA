import random

import matplotlib.pyplot as plt
import numpy as np
from ray.rllib.env import MultiAgentEnv

ACTIONS = {'MOVE_LEFT': [0, -1],
           'MOVE_RIGHT': [0, 1],
           'MOVE_UP': [-1, 0],
           'MOVE_DOWN': [1, 0],
           'STAY': [0, 0],
           'TURN_CLOCKWISE': [[0, 1], [-1, 0]],
           'TURN_COUNTERCLOCKWISE': [[0, -1], [1, 0]]}

ORIENTATIONS = {'LEFT': [0, -1],
                'RIGHT': [0, 1],
                'UP': [-1, 0],
                'DOWN': [1, 0]}

DEFAULT_COLOURS = {' ': [0, 0, 0],
                   '0': [0, 0, 0],
                   '': [180, 180, 180],
                   '@': [180, 180, 180],
                   'A': [0, 255, 0],
                   'F': [255, 255, 0],
                   'P': [159, 67, 255],

                   '1': [159, 67, 255],
                   '2': [2, 81, 154],
                   '3': [204, 0, 204],
                   '4': [216, 30, 54],
                   '5': [254, 151, 0],
                   '6': [100, 255, 255],
                   '7': [99, 99, 255],
                   '8': [250, 204, 255],
                   '9': [238, 223, 16]}


class MapEnv(MultiAgentEnv):

    def __init__(self, ascii_map, num_agents=1, render=True, color_map=None,
                 shuffle_spawn=True, random_orientation=True,
                 beam_width=3):
        self.num_agents = num_agents
        self.base_map = self.ascii_to_numpy(ascii_map)
        self.world_map = np.full((len(self.base_map), len(self.base_map[0])), ' ')
        self.beam_pos = []
        self.shuffle_spawn = shuffle_spawn
        self.random_orientation = random_orientation
        assert beam_width % 2 == 1
        self.beam_shift = int((beam_width - 1)/2)

        self.agents = {}

        self.pos_dict = {}
        self.color_map = color_map if color_map is not None else DEFAULT_COLOURS
        self.spawn_points = []

        self.wall_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == 'P':
                    self.spawn_points.append([row, col])
                elif self.base_map[row, col] == '@':
                    self.wall_points.append([row, col])
        self.setup_agents()

    def custom_reset(self):
        pass

    def custom_action(self, agent, action):
        pass

    def custom_map_update(self):
        pass

    def setup_agents(self):
        raise NotImplementedError

    def ascii_to_numpy(self, ascii_list):
        arr = np.full((len(ascii_list), len(ascii_list[0])), ' ')
        for row in range(arr.shape[0]):
            for col in range(arr.shape[1]):
                arr[row, col] = ascii_list[row][col]
        return arr

    def process_rgb_arr(self, rgb_arr):

        pos = [(rgb_arr.shape[0] - 1)//2, (rgb_arr.shape[1] - 1)//2]
        rgb_arr[pos[0]][pos[1]] = np.array(DEFAULT_COLOURS['1'])

        return rgb_arr

    def step(self, actions):
        self.beam_pos = []
        agent_actions = {}
        for agent_id, action in actions.items():
            agent_action = self.agents[agent_id].action_map(action)
            agent_actions[agent_id] = agent_action

        self.update_moves(agent_actions)

        for agent in self.agents.values():
            pos = agent.get_pos()
            new_char = agent.consume(self.world_map[pos[0], pos[1]])
            self.world_map[pos[0], pos[1]] = new_char

        n_cleaned_each_agent = self.update_custom_moves(agent_actions)

        self.custom_map_update()

        map_with_agents = self.get_map_with_agents()

        observations = []
        rewards = {}
        dones = {}
        info = {}
        for agent in self.agents.values():
            agent.grid = map_with_agents
            rgb_arr = self.map_to_colors(agent.get_state(), self.color_map)
            rgb_arr = self.rotate_view(agent.orientation, rgb_arr)
            rgb_arr = self.process_rgb_arr(rgb_arr)
            observations.append(rgb_arr.reshape((rgb_arr.shape[0] * rgb_arr.shape[1] * rgb_arr.shape[2])))
            rewards[agent.agent_id] = agent.compute_reward()
            dones[agent.agent_id] = agent.get_done()
        dones["__all__"] = np.any(list(dones.values()))
        info['n_cleaned_each_agent'] = n_cleaned_each_agent
        return observations, rewards, dones, info

    def get_agent_obs(self):
        observations = []
        for agent in self.agents.values():
            rgb_arr = self.map_to_colors(agent.get_state(), self.color_map)
            rgb_arr = self.rotate_view(agent.orientation, rgb_arr)
            rgb_arr = self.process_rgb_arr(rgb_arr)
            observations.append(rgb_arr.reshape((rgb_arr.shape[0] * rgb_arr.shape[1] * rgb_arr.shape[2])))
        return observations


    def reset(self):
        self.beam_pos = []
        self.agents = {}
        self.setup_agents()
        self.reset_map()
        self.custom_map_update()

        map_with_agents = self.get_map_with_agents()

        observations = []
        for agent in self.agents.values():
            agent.grid = map_with_agents
            rgb_arr = self.map_to_colors(agent.get_state(), self.color_map)
            rgb_arr = self.rotate_view(agent.orientation, rgb_arr)
            rgb_arr = self.process_rgb_arr(rgb_arr)
            observations.append(rgb_arr.reshape((rgb_arr.shape[0] * rgb_arr.shape[1] * rgb_arr.shape[2])))
        return observations

    @property
    def agent_pos(self):
        return [agent.get_pos().tolist() for agent in self.agents.values()]

    def get_map_with_agents(self):
        grid = np.copy(self.world_map)

        for agent_id, agent in self.agents.items():
            char_id = str(int(agent_id[-1]) + 1)

            if not(agent.pos[0] >= 0 and agent.pos[0] < grid.shape[0] and
                   agent.pos[1] >= 0 and agent.pos[1] < grid.shape[1]):
                continue

            grid[agent.pos[0], agent.pos[1]] = char_id

        for beam_pos in self.beam_pos:
            grid[beam_pos[0], beam_pos[1]] = beam_pos[2]

        return grid

    def map_to_colors(self, map=None, color_map=None):
        if map is None:
            map = self.get_map_with_agents()
        if color_map is None:
            color_map = self.color_map

        rgb_arr = np.zeros((map.shape[0], map.shape[1], 3), dtype=int)
        for row_elem in range(map.shape[0]):
            for col_elem in range(map.shape[1]):
                rgb_arr[row_elem, col_elem, :] = color_map[map[row_elem, col_elem]]

        return rgb_arr

    def render(self, filename=None):
        map_with_agents = self.get_map_with_agents()

        rgb_arr = self.map_to_colors(map_with_agents)
        plt.imshow(rgb_arr, interpolation='nearest')
        if filename is None:
            plt.draw()
            plt.pause(0.1)
        else:
            plt.savefig(filename)

    def update_moves(self, agent_actions):
        reserved_slots = []
        for agent_id, action in agent_actions.items():
            agent = self.agents[agent_id]
            selected_action = ACTIONS[action]
            if 'MOVE' in action or 'STAY' in action:
                rot_action = self.rotate_action(selected_action, agent.get_orientation())
                new_pos = agent.get_pos() + rot_action
                new_pos = agent.return_valid_pos(new_pos)
                reserved_slots.append((*new_pos, 'P', agent_id))
            elif 'TURN' in action:
                new_rot = self.update_rotation(action, agent.get_orientation())
                agent.update_agent_rot(new_rot)

        agent_by_pos = {tuple(agent.get_pos()): agent.agent_id for agent in self.agents.values()}

        agent_moves = {}

        move_slots = []
        agent_to_slot = []

        for slot in reserved_slots:
            row, col = slot[0], slot[1]
            if slot[2] == 'P':
                agent_id = slot[3]
                agent_moves[agent_id] = [row, col]
                move_slots.append([row, col])
                agent_to_slot.append(agent_id)

        if len(agent_to_slot) > 0:

            shuffle_list = list(zip(agent_to_slot, move_slots))
            np.random.shuffle(shuffle_list)
            agent_to_slot, move_slots = zip(*shuffle_list)
            unique_move, indices, return_count = np.unique(move_slots, return_index=True,
                                                           return_counts=True, axis=0)
            search_list = np.array(move_slots)

            if np.any(return_count > 1):
                for move, index, count in zip(unique_move, indices, return_count):
                    if count > 1:
                        conflict_indices = np.where((search_list == move).all(axis=1))[0]
                        all_agents_id = [agent_to_slot[i] for i in conflict_indices]
                        conflict_cell_free = True
                        for agent_id in all_agents_id:
                            moves_copy = agent_moves.copy()
                            if move.tolist() in self.agent_pos:
                                conflicting_agent_id = agent_by_pos[tuple(move)]
                                curr_pos = self.agents[agent_id].get_pos().tolist()
                                curr_conflict_pos = self.agents[conflicting_agent_id]. \
                                    get_pos().tolist()
                                conflict_move = agent_moves.get(conflicting_agent_id,
                                                                curr_conflict_pos)
                                if agent_id == conflicting_agent_id:
                                    conflict_cell_free = False
                                elif conflicting_agent_id not in moves_copy.keys() or \
                                        curr_conflict_pos == conflict_move:
                                    conflict_cell_free = False

                                elif conflicting_agent_id in moves_copy.keys():
                                    if agent_moves[conflicting_agent_id] == curr_pos and \
                                            move.tolist() == self.agents[conflicting_agent_id] \
                                            .get_pos().tolist():
                                        conflict_cell_free = False

                        if conflict_cell_free:
                            self.agents[agent_to_slot[index]].update_agent_pos(move)
                            agent_by_pos = {tuple(agent.get_pos()):
                                            agent.agent_id for agent in self.agents.values()}
                        remove_indices = np.where((search_list == move).all(axis=1))[0]
                        all_agents_id = [agent_to_slot[i] for i in remove_indices]
                        for agent_id in all_agents_id:
                            agent_moves[agent_id] = self.agents[agent_id].get_pos().tolist()

            while len(agent_moves.items()) > 0:
                agent_by_pos = {tuple(agent.get_pos()):
                                agent.agent_id for agent in self.agents.values()}
                num_moves = len(agent_moves.items())
                moves_copy = agent_moves.copy()
                del_keys = []
                for agent_id, move in moves_copy.items():
                    if agent_id in del_keys:
                        continue
                    if move in self.agent_pos:
                        conflicting_agent_id = agent_by_pos[tuple(move)]
                        curr_pos = self.agents[agent_id].get_pos().tolist()
                        curr_conflict_pos = self.agents[conflicting_agent_id].get_pos().tolist()
                        conflict_move = agent_moves.get(conflicting_agent_id, curr_conflict_pos)
                        if agent_id == conflicting_agent_id:
                            del agent_moves[agent_id]
                            del_keys.append(agent_id)
                        elif conflicting_agent_id not in moves_copy.keys() or \
                                curr_conflict_pos == conflict_move:
                            del agent_moves[agent_id]
                            del_keys.append(agent_id)
                        elif conflicting_agent_id in moves_copy.keys():
                            if agent_moves[conflicting_agent_id] == curr_pos and \
                                    move == self.agents[conflicting_agent_id].get_pos().tolist():
                                del agent_moves[conflicting_agent_id]
                                del agent_moves[agent_id]
                                del_keys.append(agent_id)
                                del_keys.append(conflicting_agent_id)
                    else:
                        self.agents[agent_id].update_agent_pos(move)
                        del agent_moves[agent_id]
                        del_keys.append(agent_id)

                if len(agent_moves) == num_moves:
                    for agent_id, move in agent_moves.items():
                        self.agents[agent_id].update_agent_pos(move)
                    break

    def update_custom_moves(self, agent_actions):
        n_cleaned_each_agent = []
        for agent_id, action in agent_actions.items():
            n_cleaned = 0
            if 'MOVE' not in action and 'STAY' not in action and 'TURN' not in action:
                agent = self.agents[agent_id]
                updates = self.custom_action(agent, action)
                if len(updates) > 0:
                    self.update_map(updates)
                    for tup in updates:
                        n_cleaned += 1 if tup[2] == 'R' else 0
            n_cleaned_each_agent.append(n_cleaned)
        return n_cleaned_each_agent

    def update_map(self, new_points):
        for i in range(len(new_points)):
            row, col, char = new_points[i]
            self.world_map[row, col] = char

    def reset_map(self):
        self.world_map = np.full((len(self.base_map), len(self.base_map[0])), ' ')
        self.build_walls()
        self.custom_reset()

    def update_map_fire(self, firing_pos, firing_orientation, fire_len, fire_char, cell_types=[],
                        update_char=[], blocking_cells='P'):
        agent_by_pos = {tuple(agent.get_pos()): agent_id for agent_id, agent in self.agents.items()}
        start_pos = np.asarray(firing_pos)
        firing_direction = ORIENTATIONS[firing_orientation]
        right_shift = self.rotate_right(firing_direction)
        if self.beam_shift == 0:
            firing_pos = [start_pos]
        else:
            firing_pos = [start_pos, start_pos + self.beam_shift*right_shift - firing_direction, start_pos - self.beam_shift*right_shift - firing_direction]
        firing_points = []
        updates = []
        for pos in firing_pos:
            next_cell = pos + firing_direction
            for i in range(fire_len):
                if self.test_if_in_bounds(next_cell) and \
                        self.world_map[next_cell[0], next_cell[1]] != '@':

                    if [next_cell[0], next_cell[1]] in self.agent_pos:
                        agent_id = agent_by_pos[(next_cell[0], next_cell[1])]
                        self.agents[agent_id].hit(fire_char)
                        firing_points.append((next_cell[0], next_cell[1], fire_char))
                        if self.world_map[next_cell[0], next_cell[1]] in cell_types:
                            type_index = cell_types.index(self.world_map[next_cell[0],
                                                                         next_cell[1]])
                            updates.append((next_cell[0], next_cell[1], update_char[type_index]))
                        break

                    if self.world_map[next_cell[0], next_cell[1]] in cell_types:
                        type_index = cell_types.index(self.world_map[next_cell[0], next_cell[1]])
                        updates.append((next_cell[0], next_cell[1], update_char[type_index]))

                    firing_points.append((next_cell[0], next_cell[1], fire_char))

                    if self.world_map[next_cell[0], next_cell[1]] in blocking_cells:
                        break

                    next_cell += firing_direction

                else:
                    break

        self.beam_pos += firing_points
        return updates

    def spawn_point(self):
        spawn_index = 0
        is_free_cell = False
        curr_agent_pos = [agent.get_pos().tolist() for agent in self.agents.values()]
        if self.shuffle_spawn:
            random.shuffle(self.spawn_points)
        for i, spawn_point in enumerate(self.spawn_points):
            if [spawn_point[0], spawn_point[1]] not in curr_agent_pos:
                spawn_index = i
                is_free_cell = True
        assert is_free_cell, 'There are not enough spawn points! Check your map?'
        return np.array(self.spawn_points[spawn_index])

    def spawn_rotation(self):
        if self.random_orientation:
            rand_int = np.random.randint(len(ORIENTATIONS.keys()))
            orientation = list(ORIENTATIONS.keys())[rand_int]
        else:
            orientation = 'UP'
        return orientation

    def rotate_view(self, orientation, view):
        if orientation == 'UP':
            return view
        elif orientation == 'LEFT':
            return np.rot90(view, k=3, axes=(0, 1))
        elif orientation == 'DOWN':
            return np.rot90(view, k=2, axes=(0, 1))
        elif orientation == 'RIGHT':
            return np.rot90(view, k=1, axes=(0, 1))
        else:
            raise ValueError('Orientation {} is not valid'.format(orientation))

    def build_walls(self):
        for i in range(len(self.wall_points)):
            row, col = self.wall_points[i]
            self.world_map[row, col] = '@'

    def rotate_action(self, action_vec, orientation):
        if orientation == 'UP':
            return action_vec
        elif orientation == 'LEFT':
            return self.rotate_left(action_vec)
        elif orientation == 'RIGHT':
            return self.rotate_right(action_vec)
        else:
            return self.rotate_left(self.rotate_left(action_vec))

    def rotate_left(self, action_vec):
        return np.dot(ACTIONS['TURN_COUNTERCLOCKWISE'], action_vec)

    def rotate_right(self, action_vec):
        return np.dot(ACTIONS['TURN_CLOCKWISE'], action_vec)

    def update_rotation(self, action, curr_orientation):
        if action == 'TURN_COUNTERCLOCKWISE':
            if curr_orientation == 'LEFT':
                return 'DOWN'
            elif curr_orientation == 'DOWN':
                return 'RIGHT'
            elif curr_orientation == 'RIGHT':
                return 'UP'
            else:
                return 'LEFT'
        else:
            if curr_orientation == 'LEFT':
                return 'UP'
            elif curr_orientation == 'UP':
                return 'RIGHT'
            elif curr_orientation == 'RIGHT':
                return 'DOWN'
            else:
                return 'LEFT'

    def test_if_in_bounds(self, pos):
        if pos[0] < 0 or pos[0] >= self.world_map.shape[0]:
            return False
        elif pos[1] < 0 or pos[1] >= self.world_map.shape[1]:
            return False
        else:
            return True
