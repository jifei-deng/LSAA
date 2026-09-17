import numpy as np
import torch


class Agents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        if args.alg.find('lsaa') > -1:
            from policy.lsaa import LSAA
            self.policy = LSAA(args)
        else:
            raise Exception("No such algorithm")
        self.args = args

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, episode_counts=None, agent_obs=None,
                      obs_act=None, gmm_manger=None, ask_budget=None, ans_budget=None):
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]

        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))
            hidden_state = self.policy.eval_hidden[:, agent_num, :]
        else:
            hidden_state = self.policy.eval_hidden[agent_num]

        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        if self.args.reuse_network:
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)
        else:
            q_value, self.policy.eval_hidden[agent_num] = self.policy.eval_rnn[agent_num](inputs, hidden_state)

        q_value[avail_actions == 0.0] = - float("inf")
        if np.random.uniform() >= epsilon:
            action = torch.argmax(q_value)
            probs = torch.softmax(q_value, dim=-1)
        else:
            if avail_actions_ind.shape[0] <= 1:
                action = avail_actions_ind[0]
                probs = np.zeros(self.args.n_actions)
                probs[action] = 1.0
            else:
                if episode_counts > self.args.start_advice and ask_budget[agent_num] > 0:
                    action, ask_budget, probs, ans_budget = self.policy.ask_advice(q_value, obs, agent_num, agent_obs,
                                                                                   ask_budget, ans_budget,
                                                                                   episode_counts, obs_act, gmm_manger)
                    if (action is None) or (action not in avail_actions_ind):
                        if np.random.uniform() < epsilon:
                            action = np.random.choice(avail_actions_ind)
                            probs = np.zeros(self.args.n_actions)
                            uniform_prob = 1.0 / len(avail_actions_ind)
                            probs[avail_actions_ind] = uniform_prob
                        else:
                            action = torch.argmax(q_value)
                            probs = torch.softmax(q_value, dim=-1)
                else:
                    if np.random.uniform() < epsilon:
                        action = np.random.choice(avail_actions_ind)
                        probs = np.zeros(self.args.n_actions)
                        uniform_prob = 1.0 / len(avail_actions_ind)
                        probs[avail_actions_ind] = uniform_prob
                    else:
                        action = torch.argmax(q_value)
                        probs = torch.softmax(q_value, dim=-1)
        return action, ask_budget, probs, ans_budget

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        if max_episode_len == 0:
            max_episode_len = self.args.episode_limit
        return max_episode_len

    def train(self, batch, train_step, episode_counts, epsilon=None):
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if episode_counts > 0 and episode_counts % self.args.save_cycle == 0:
            self.policy.save_model(train_step, episode_counts)
