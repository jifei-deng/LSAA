import numpy as np
import time


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon

        print('Init RolloutWorker')

    def generate_episode(self, episode_counts=None, episode_num=None, agent_obs=None, ask_budget=None, ans_budget=None, evaluate=False, obs_act=None, gmm_manger=None):
        if evaluate and episode_num == 0:
            self.env.close()

        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []

        self.env.reset()
        terminated = False

        step = 0
        episode_reward = 0
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)

        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        while not terminated and step < self.episode_limit:
            if self.args.render:
                time.sleep(0.1)
                self.env.render()

            obs = self.env.get_obs()
            state = self.env.get_state()
            if not evaluate:
                for i in range(self.n_agents):

                    if tuple(obs[i]) in agent_obs[i]:
                        agent_obs[i][tuple(obs[i])] = agent_obs[i][tuple(obs[i])] + 1
                    else:
                        agent_obs[i][tuple(obs[i])] = 1

            actions, avail_actions, actions_onehot = [], [], []

            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)

                action, ask_budget, probs, ans_budget = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                                                  avail_action, epsilon, episode_counts, agent_obs,
                                                                                  obs_act, gmm_manger, ask_budget, ans_budget)

                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(np.int(action))
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

                if not evaluate:
                    if tuple(obs[agent_id]) in obs_act[agent_id]:
                        obs_act[agent_id][tuple(obs[agent_id])].append(probs)
                    else:
                        obs_act[agent_id][tuple(obs[agent_id])] = [probs]

            next_obs, reward, terminated, info = self.env.step(actions)

            terminated = True if False not in terminated else False
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            terminate.append([terminated])
            padded.append([0.])
            if self.args.individual_rewards:
                r.append(reward)
                episode_reward += sum(reward)
            else:
                reward = sum(reward)
                r.append([reward])
                episode_reward += reward
            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        if self.args.render:
            time.sleep(0.1)
            self.env.render()

        obs = self.env.get_obs()
        state = self.env.get_state()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            if self.args.individual_rewards:
                r.append(np.zeros(self.n_agents))
            else:
                r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon

        if evaluate and episode_num == self.args.evaluate_episode - 1:
            self.env.close()
        return episode, episode_reward, r, step, agent_obs, ask_budget, obs_act, ans_budget
