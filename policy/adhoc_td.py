import torch
import os
from network.base_net import RNN
import torch.nn.functional as F
import math
import random
from collections import Counter
import numpy as np


class AdhocTD:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        input_shape = self.obs_shape
        self.args = args
        if not (args.evaluate and args.load_model):
            self.model_dir = args.model_dir
        else:
            self.model_dir = args.pkl_dir


        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents


            self.eval_rnn = RNN(input_shape, args)
            self.target_rnn = RNN(input_shape, args)
            if self.args.cuda:
                self.eval_rnn.cuda()
                self.target_rnn.cuda()

            if self.args.load_model:
                if os.path.exists(self.model_dir + 'rnn_net_params.pkl'):
                    path_rnn = self.model_dir + 'rnn_net_params.pkl'
                    map_location = 'cuda:0' if self.args.cuda else 'cpu'
                    self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                    print('Successfully load the model: {}'.format(path_rnn))
                else:
                    raise Exception("No model!")

            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())

            self.eval_parameters = list(self.eval_rnn.parameters())
            if args.optimizer == "RMS":
                self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr, eps=args.eps)

            self.eval_hidden = None
            self.target_hidden = None
        else:
            self.eval_rnn, self.target_rnn, self.eval_hidden, self.target_hidden = [], [], [], []
            self.eval_parameters, self.optimizer = [], []
            for i in range(self.n_agents):
                self.eval_rnn.append(RNN(input_shape, args))
                self.target_rnn.append(RNN(input_shape, args))
                self.eval_hidden.append(None)
                self.target_hidden.append(None)
                if self.args.cuda:
                    self.eval_rnn[i].cuda()
                    self.target_rnn[i].cuda()
                if self.args.load_model:
                    if os.path.exists(self.model_dir + 'rnn_net_params_' + str(i) + '.pkl'):
                        path_rnn = self.model_dir + 'rnn_net_params_' + str(i) + '.pkl'
                        map_location = 'cuda:0' if self.args.cuda else 'cpu'
                        self.eval_rnn[i].load_state_dict(torch.load(path_rnn, map_location=map_location))
                        print('Successfully load the model: {}'.format(path_rnn))
                    else:
                        raise Exception("No model!")

                self.target_rnn[i].load_state_dict(self.eval_rnn[i].state_dict())
                self.eval_parameters.append(list(self.eval_rnn[i].parameters()))
                if args.optimizer == "RMS":
                    self.optimizer.append(torch.optim.RMSprop(self.eval_parameters[i], lr=args.lr, eps=args.eps))
        random.seed(args.seed)
        print('Init alg Adhoc_TD')

    def learn(self, batch, max_episode_len, train_step, epsilon=None):

        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        u, avail_u, avail_u_next, terminated = batch['u'], batch['avail_u'], batch['avail_u_next'], \
                                               batch['terminated'].repeat(1, 1, self.n_agents)
        if self.args.individual_rewards:
            r = batch['r']
        else:
            r = batch['r'].repeat(1, 1, self.n_agents)
        mask = (1 - batch["padded"].float()).repeat(1, 1, self.n_agents)


        q_evals, q_targets = self.get_q_values(batch, max_episode_len)

        if self.args.cuda:
            u = u.cuda()
            r = r.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
        if self.args.reuse_network:

            q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)

            q_targets[avail_u_next == 0.0] = - 9999999
            q_targets = q_targets.max(dim=3)[0]

            targets = r + self.args.gamma * q_targets * (1 - terminated)
            td_error = (q_evals - targets.detach())
            masked_td_error = mask * td_error


            loss = (masked_td_error ** 2).sum() / mask.sum()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
            self.optimizer.step()

            if train_step > 0 and train_step % self.args.target_update_cycle == 0:
                self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            return loss
        else:
            u = u.permute(2, 0, 1, 3)
            avail_u_next = avail_u_next.permute(2, 0, 1, 3)
            r = r.permute(2, 0, 1)
            terminated = terminated.permute(2, 0, 1)
            mask = mask.permute(2, 0, 1)
            all_loss = []
            for i in range(self.n_agents):



                q_evals[i] = torch.gather(q_evals[i], dim=2, index=u[i]).squeeze(2)



                q_targets[i][avail_u_next[i] == 0.0] = - 9999999
                q_targets[i] = q_targets[i].max(dim=2)[0]

                targets = r[i] + self.args.gamma * q_targets[i] * (1 - terminated[i])
                td_error = (q_evals[i] - targets.detach())
                masked_td_error = mask[i] * td_error


                loss = (masked_td_error ** 2).sum() / mask[i].sum()
                all_loss.append(loss)

                self.optimizer[i].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.eval_parameters[i], self.args.grad_norm_clip)
                self.optimizer[i].step()

            if train_step > 0 and train_step % self.args.target_update_cycle == 0:
                for i in range(self.n_agents):
                    self.target_rnn[i].load_state_dict(self.eval_rnn[i].state_dict())
            return sum(all_loss)

    def _get_inputs(self, batch, transition_idx):

        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)

        if self.args.last_action:
            if transition_idx == 0:
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
            inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        else:
            inputs = torch.cat(inputs, dim=2).permute(1, 0, 2)
            inputs_next = torch.cat(inputs_next, dim=2).permute(1, 0, 2)
        return inputs, inputs_next

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        if self.args.reuse_network:
            for transition_idx in range(max_episode_len):
                inputs, inputs_next = self._get_inputs(batch, transition_idx)
                if self.args.cuda:
                    inputs = inputs.cuda()
                    inputs_next = inputs_next.cuda()
                    self.eval_hidden = self.eval_hidden.cuda()
                    self.target_hidden = self.target_hidden.cuda()
                q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
                q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

                q_eval = q_eval.view(episode_num, self.n_agents, -1)
                q_target = q_target.view(episode_num, self.n_agents, -1)
                q_evals.append(q_eval)
                q_targets.append(q_target)


            q_evals = torch.stack(q_evals, dim=1)
            q_targets = torch.stack(q_targets, dim=1)
        else:
            for i in range(self.n_agents):
                q_eval_i = []
                q_target_i = []
                for transition_idx in range(max_episode_len):
                    inputs, inputs_next = self._get_inputs(batch, transition_idx)
                    if self.args.cuda:
                        inputs = inputs.cuda()
                        inputs_next = inputs_next.cuda()
                        self.eval_hidden[i] = self.eval_hidden[i].cuda()
                        self.target_hidden[i] = self.target_hidden[i].cuda()
                    q, self.eval_hidden[i] = self.eval_rnn[i](inputs[i], self.eval_hidden[i])
                    q_t, self.target_hidden[i] = self.target_rnn[i](inputs_next[i], self.target_hidden[i])
                    q_eval_i.append(q)
                    q_target_i.append(q_t)
                q_eval_i = torch.stack(q_eval_i, dim=1)
                q_target_i = torch.stack(q_target_i, dim=1)
                q_evals.append(q_eval_i)
                q_targets.append(q_target_i)

        return q_evals, q_targets

    def init_hidden(self, episode_num):

        if self.args.reuse_network:
            self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
            self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        else:
            for i in range(self.n_agents):
                self.eval_hidden[i] = torch.zeros((episode_num, self.args.rnn_hidden_dim))
                self.target_hidden[i] = torch.zeros((episode_num, self.args.rnn_hidden_dim))

    def save_model(self, train_step, episode_counts):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if self.args.reuse_network:
            torch.save(self.eval_rnn.state_dict(), f"{self.model_dir}/{num}({episode_counts})_rnn_net_params.pkl")
        else:
            for i in range(self.n_agents):
                torch.save(self.eval_rnn[i].state_dict(), f"{self.model_dir}/{num}({episode_counts})_rnn_net_params_{str(i)}.pkl")

    def select_from_advice(self, advised_list):
        if len(advised_list) == 0:
            return None
        elif len(advised_list) == 1:
            return advised_list[0]
        else:
            count = Counter(advised_list).most_common()
            action_index = []
            most_times = count[0][1]
            for tuple in count:
                if tuple[1] < most_times:
                    break
                if tuple[1] == most_times:
                    action_index.append(tuple[0])
            action = random.choices(action_index, k=1)[0]
        return action

    def ask_advice(self, out, hidden_state, obs, last_action, i, agent_obs, epi_obs, agent_budge, episode_counts):
        out = F.softmax(out, dim=-1)
        agent_id = np.zeros(self.n_agents)
        agent_id[i] = 1.


        if random.random() < math.pow((1 + self.args.variable_a), -math.sqrt(agent_obs[i][tuple(obs)])) and obs not in np.array(epi_obs[i]):
            agent_budge[i] = agent_budge[i] - 1
            s, s_key = [], []
            for j in range(self.n_agents):
                s.append(obs)
                s_key.append(obs)
                if self.args.last_action:
                    s[j] = np.hstack((s[j], last_action))
                if self.args.reuse_network:
                    s[j] = np.hstack((s[j], np.eye(self.n_agents)[j]))
            obs_adv = torch.Tensor(np.array(s))
            if self.args.cuda:
                obs_adv = obs_adv.cuda()
                out_adv = torch.zeros([self.n_agents, self.n_actions]).cuda()
            else:
                out_adv = torch.zeros([self.n_agents, self.n_actions])
            if self.args.reuse_network:
                out_adv, _ = self.eval_rnn(obs_adv, hidden_state.repeat(self.n_agents, 1))
            else:
                for advisor_i in range(self.n_agents):
                    out_adv[advisor_i], _ = self.eval_rnn[advisor_i](obs_adv[advisor_i].unsqueeze(0), hidden_state)
            out_adv = F.softmax(out_adv, dim=-1).detach()

            advised_actions = []
            for j in range(self.n_agents):
                if i == j:
                    continue
                if self.args.alg.find('v1'):
                    advised_actions.append(out_adv[j, :].argmax().item())
                else:
                    difQ = math.fabs(out_adv[j, :].max() - out_adv[j, :].min())
                    numberVisits = 0 if tuple(s_key[j]) not in agent_obs[j] else agent_obs[j][tuple(s_key[j])]
                    value = (math.sqrt(numberVisits) * difQ)
                    prob = 1 - (math.pow((1 + self.args.variable_g), -value))

                    if random.random() < prob:
                        advised_actions.append(out_adv[j, :].argmax().item())

            action = self.select_from_advice(advised_actions)
            if action is None:
                agent_budge[i] = agent_budge[i] + 1
            return action, agent_budge

        action = None
        return action, agent_budge
