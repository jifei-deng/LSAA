import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F


class Agents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        if args.alg == 'vdn':
            from policy.vdn import VDN
            self.policy = VDN(args)
        elif args.alg == 'iql':
            from policy.iql import IQL
            self.policy = IQL(args)
        elif args.alg.find('lsaa') > -1:
            from policy.lsaa import LSAA
            self.policy = LSAA(args)
        elif args.alg.find('adhoc_td') > -1:
            from policy.adhoc_td import AdhocTD
            self.policy = AdhocTD(args)
        elif args.alg == 'qmix':
            from policy.qmix import QMIX
            self.policy = QMIX(args)
        elif args.alg == 'coma':
            from policy.coma import COMA
            self.policy = COMA(args)
        elif args.alg == 'qtran_alt':
            from policy.qtran_alt import QtranAlt
            self.policy = QtranAlt(args)
        elif args.alg == 'qtran_base':
            from policy.qtran_base import QtranBase
            self.policy = QtranBase(args)
        elif args.alg == 'maven':
            from policy.maven import MAVEN
            self.policy = MAVEN(args)
        elif args.alg == 'central_v':
            from policy.central_v import CentralV
            self.policy = CentralV(args)
        elif args.alg == 'reinforce':
            from policy.reinforce import Reinforce
            self.policy = Reinforce(args)
        else:
            raise Exception("No such algorithm")
        self.args = args

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, episode_counts=None, agent_obs=None,
                      obs_act=None, gmm_manger=None, epi_obs=None, ask_budget=None, ans_budget=None, maven_z=None, evaluate=False):
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

            if self.args.alg == 'maven':
                maven_z = torch.tensor(maven_z, dtype=torch.float32).unsqueeze(0)
                if self.args.cuda:
                    maven_z = maven_z.cuda()
                q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state, maven_z)

            else:
                q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)


        else:
            if self.args.alg == 'maven':
                maven_z = torch.tensor(maven_z, dtype=torch.float32).unsqueeze(0)
                if self.args.cuda:
                    maven_z = maven_z.cuda()
                q_value, self.policy.eval_hidden[agent_num] = self.policy.eval_rnn[agent_num](inputs, hidden_state, maven_z)
            else:
                q_value, self.policy.eval_hidden[agent_num] = self.policy.eval_rnn[agent_num](inputs, hidden_state)


        if self.args.alg == 'coma' or self.args.alg == 'central_v' or self.args.alg == 'reinforce':
            action = self._choose_action_from_softmax(q_value.cpu(), avail_actions, epsilon, evaluate)

        if self.args.alg.find('lsaa') > -1:
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




                        action, ask_budget, probs, ans_budget = self.policy.ask_advice_gmm_3_lsaa_1(q_value, hidden_state,
                                                                    obs, last_action, agent_num,
                                                                    agent_obs, epi_obs, ask_budget, ans_budget,
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

    def _choose_action_from_softmax(self, inputs, avail_actions, epsilon, evaluate=False):

        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])

        prob = F.softmax(inputs, dim=-1)

        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0



        if epsilon == 0 and evaluate:
            action = torch.argmax(prob)
        else:
            action = Categorical(prob).sample().long()
        return action

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
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if episode_counts > 0 and episode_counts % self.args.save_cycle == 0:
            self.policy.save_model(train_step, episode_counts)
