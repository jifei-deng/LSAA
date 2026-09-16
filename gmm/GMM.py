from collections import defaultdict
from sklearn.mixture import GaussianMixture
import numpy as np

class GMMManager:
    def __init__(self, n_agents, max_obs_per_agent=None):
        self.n_agents = n_agents
        self.gmms = [defaultdict(dict) for _ in range(n_agents)]
        self.max_obs_per_agent = max_obs_per_agent

    def train_gmm(self, agent_id, obs, data, label='positive', n_components=4):

        obs_key = tuple(obs)
        if self.max_obs_per_agent is not None and len(self.gmms[agent_id]) >= self.max_obs_per_agent:

            print(f"[GMMManager] Agent {agent_id} exceeds max obs limit. Skipping training.")
            return

        gmm = GaussianMixture(n_components=n_components, covariance_type='full')
        gmm.fit(data)
        self.gmms[agent_id][obs_key][label] = gmm

    def sample(self, agent_id, obs, label='positive'):

        obs_key = tuple(obs)
        gmm = self.gmms[agent_id].get(obs_key, {}).get(label, None)
        if gmm is None:
            return None
        sample = gmm.sample(1)[0]
        return sample[0]
