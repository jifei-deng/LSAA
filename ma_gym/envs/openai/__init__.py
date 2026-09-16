import gym

from ..utils.action_space import MultiAgentActionSpace
from ..utils.observation_space import MultiAgentObservationSpace


class MultiAgentWrapper(gym.Wrapper):


    def __init__(self, name):
        super().__init__(gym.make(name))
        self.n_agents = 1
        self._step_count = None
        self._total_episode_reward = None
        self._agent_dones = [None for _ in range(self.n_agents)]

        self.action_space = MultiAgentActionSpace([self.env.action_space])
        self.observation_space = MultiAgentObservationSpace([self.env.observation_space])

    def step(self, action_n):
        assert (self._step_count is not None), \
            "Call reset before using step method."

        self._step_count += 1
        assert len(action_n) == self.n_agents

        action = action_n[0]
        obs, reward, done, info = self.env.step(action)






        if self.env._elapsed_steps == (self.env._max_episode_steps - 1):
            done = True

        self._total_episode_reward[0] += reward

        return [obs], [reward], [done], info

    def reset(self):
        self._step_count = 0
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self._agent_dones = [False for _ in range(self.n_agents)]

        obs = self.env.reset()
        return [obs]
