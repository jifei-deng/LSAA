import torch
import os
from network.base_net import RNN
import torch.nn.functional as F
import math
import random
import numpy as np


class LSAA:
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
        print('Init alg LSAA')

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
            torch.save(self.eval_rnn.state_dict(),  f"{self.model_dir}/{num}({episode_counts})_rnn_net_params.pkl")
        else:
            for i in range(self.n_agents):
                torch.save(self.eval_rnn[i].state_dict(), f"{self.model_dir}/{num}({episode_counts})_rnn_net_params_{str(i)}.pkl")

    def ask_advice(self, q_value, obs, i, agent_obs, ask_budget, ans_budget, episode_counts, obs_act, gmm_manger):
        out = F.softmax(q_value, dim=-1)
        obs_key = tuple(obs)

        policy_entropy = -torch.sum(out * torch.log(out + 1e-8)).item()
        entropy_bonus = np.exp(-policy_entropy)

        base_prob = math.pow((1 + self.args.variable_a), -math.sqrt(agent_obs[i][obs_key]))
        adjusted_prob = base_prob * (1 + self.args.beta * entropy_bonus)

        if random.random() < adjusted_prob:

            current_policy_np = out.detach().cpu().numpy()[0]
            q_np = q_value.detach().cpu().numpy()[0]
            q_np = np.where(np.isfinite(q_np), q_np, 0.0)
            value_i = np.sum(current_policy_np * q_np)
            log_n_actions = math.log(self.n_actions)

            gmm_advs_pos, gmm_advs_neg = [], []

            for g in range(self.n_agents):
                if i == g or obs_key not in obs_act[g] or ans_budget[g] < 1:
                    continue

                all_exp = obs_act[g][obs_key]
                all_exp_clean = self.clean_and_stack_gmm_data(all_exp)

                if all_exp_clean is None or len(all_exp_clean) < 5:
                    continue

                ans_budget[g] -= 1

                for exp in all_exp_clean:
                    kl_div = np.sum(exp * np.log((exp + 1e-8) / (current_policy_np + 1e-8))) / log_n_actions
                    value_gap = np.sum(exp * q_np) - value_i

                    exp_max_action = exp.argmax()

                    if (exp[exp_max_action] > current_policy_np[exp_max_action] and
                            kl_div < self.args.tau_pos and value_gap > 0):
                        gmm_advs_pos.append(self._to_logits(exp))
                    elif (exp[exp_max_action] < current_policy_np[exp_max_action] and
                          kl_div > self.args.tau_neg and value_gap < 0):
                        gmm_advs_neg.append(self._to_logits(exp))

            n_max = self.args.n_max_per_state
            gmm_advs_pos = gmm_advs_pos[-n_max:]
            gmm_advs_neg = gmm_advs_neg[-n_max:]

            pos_logits, neg_logits = None, None
            pos_advice, neg_advice = None, None
            pos_confidence, neg_confidence = 0, 0

            min_samples = self.args.n_min_per_state

            if len(gmm_advs_pos) >= min_samples:
                pos_data = np.array(gmm_advs_pos)
                n_components = self._num_components(len(pos_data))
                gmm_manger.train_gmm(i, obs, pos_data, label='positive',
                                     n_components=n_components)
                pos_logits = gmm_manger.sample(i, obs, label='positive')
                pos_advice = self._softmax(pos_logits)
                pos_confidence = self._compute_confidence(pos_advice)

            if len(gmm_advs_neg) >= min_samples:
                neg_data = np.array(gmm_advs_neg)
                n_components = self._num_components(len(neg_data))
                gmm_manger.train_gmm(i, obs, neg_data, label='negative',
                                     n_components=n_components)
                neg_logits = gmm_manger.sample(i, obs, label='negative')
                neg_advice = self._softmax(neg_logits)
                neg_confidence = self._compute_confidence(neg_advice)

            if pos_confidence > 0 or neg_confidence > 0:
                print("LSAA advice! Episode:", episode_counts, "Confidence:", pos_confidence, neg_confidence)

                z_i = self._to_logits(current_policy_np)

                scores = {}
                if pos_advice is not None:
                    scores['pos'] = pos_confidence * math.sqrt(min(len(gmm_advs_pos), n_max))
                if neg_advice is not None:
                    scores['neg'] = neg_confidence * math.sqrt(min(len(gmm_advs_neg), n_max))
                max_score = max(scores.values())
                exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
                score_sum = sum(exp_scores.values())
                omega_pos = exp_scores.get('pos', 0.0) / score_sum
                omega_neg = exp_scores.get('neg', 0.0) / score_sum

                shift = np.zeros(self.n_actions)
                get_advice = False

                if pos_advice is not None:
                    a_pos = int(pos_advice.argmax())
                    if pos_advice[a_pos] > current_policy_np[a_pos]:
                        shift[a_pos] += omega_pos * (pos_logits[a_pos] - z_i[a_pos])
                        get_advice = True

                if neg_advice is not None:
                    a_neg = int(neg_advice.argmin())
                    if neg_advice[a_neg] < current_policy_np[a_neg]:
                        shift[a_neg] += omega_neg * (neg_logits[a_neg] - z_i[a_neg])
                        get_advice = True

                if get_advice:
                    z_fused = z_i + self.args.lam * shift
                    fused_policy_tensor = torch.tensor(z_fused, dtype=torch.float32).unsqueeze(0)
                    if self.args.cuda:
                        fused_policy_tensor = fused_policy_tensor.cuda()

                    fused_policy_tensor = F.softmax(fused_policy_tensor / self.args.temperature, dim=-1)

                    action = random.choices([_ for _ in range(self.n_actions)],
                                            weights=fused_policy_tensor[0].cpu().numpy().tolist(), k=1)[0]

                    ask_budget[i] -= 1
                    return action, ask_budget, fused_policy_tensor, ans_budget

            else:
                return None, ask_budget, None, ans_budget

        return None, ask_budget, None, ans_budget

    def _num_components(self, n_samples):
        return min(self.args.k_max, max(self.args.k_min, n_samples // self.args.rho))

    def _to_logits(self, probs):
        logits = np.log(np.clip(probs, 0.0, None) + 1e-8)
        return logits - logits.mean()

    def _softmax(self, logits):
        shifted = logits - np.max(logits)
        exp_logits = np.exp(shifted)
        return exp_logits / exp_logits.sum()

    def _compute_confidence(self, policy):
        if policy is None:
            return 0.0

        entropy = -np.sum(policy * np.log(policy + 1e-8))
        max_entropy = np.log(len(policy))
        confidence = 1.0 - entropy / max_entropy

        return confidence

    def clean_and_stack_gmm_data(self, gmm_data_raw):
        cleaned = []
        for x in gmm_data_raw:
            try:
                if isinstance(x, torch.Tensor):
                    x = x.detach().cpu().numpy()
                x = np.array(x).squeeze()
                if not np.any(np.isnan(x)) and not np.any(np.isinf(x)) and np.all(np.abs(x) < 1e8):
                    cleaned.append(x)
            except Exception:
                continue

        if len(cleaned) < 1:
            return None

        try:
            return np.stack(cleaned)
        except ValueError:
            return None
