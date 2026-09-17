import numpy as np
import torch
from common.rollout import RolloutWorker
from agent.agent import Agents
from common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
import datetime
import csv
from gmm.GMM import GMMManager
from collections import defaultdict


class Runner:
    def __init__(self, env, args):
        self.env = env

        self.agents = Agents(args)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)

        if not args.evaluate:
            self.buffer = ReplayBuffer(args)
        self.args = args
        self.episode_rewards = []

    def run(self, num):
        episode_counts, time_steps, train_steps, evaluate_steps = 0, 0, 0, -1
        with open(self.args.train_csv_file, 'w', newline='') as f_train:
            writer = csv.writer(f_train)
            writer.writerow(['episode', 'episode_reward'])
        with open(self.args.eval_csv_file, 'w', newline='') as f_eval:
            writer = csv.writer(f_eval)
            writer.writerow(['episode', 'episode_reward'])

        agent_obs = []
        for i in range(self.args.n_agents):
            agent_obs.append({})

        obs_act = [defaultdict(list) for _ in range(self.args.n_agents)]
        gmm_manger = GMMManager(self.args.n_agents)

        ask_budget, ans_budget = [], []
        for i in range(self.args.n_agents):
            ask_budget.append(self.args.consult_budget)
            ans_budget.append(self.args.respond_budget)

        while episode_counts < self.args.max_episodes + 1:
            with torch.no_grad():
                if episode_counts // self.args.evaluate_cycle > evaluate_steps:
                    episode_reward = self.evaluate()
                    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + " >> {}/{} episodes , avg evaluate score : {:.3f}, agent_budget : {}"
                        .format(episode_counts, self.args.max_episodes, episode_reward, ask_budget))

                    self.episode_rewards.append(episode_reward)
                    self.plt(num)
                    evaluate_steps += 1

                    with open(self.args.eval_csv_file, 'a+', newline='') as f_eval:
                        writer = csv.writer(f_eval)
                        writer.writerow([episode_counts, episode_reward])

                episodes = []
                for episode_idx in range(self.args.n_episodes):
                    episode, er, individual_r, steps, agent_obs, ask_budget, obs_act, ans_budget = self.rolloutWorker.generate_episode(
                        episode_counts, episode_idx, agent_obs, ask_budget, ans_budget, evaluate=self.args.evaluate,
                        obs_act=obs_act, gmm_manger=gmm_manger)
                    with open(self.args.train_csv_file, 'a+', newline='') as f_train:
                        writer = csv.writer(f_train)
                        writer.writerow([episode_counts, er]+ask_budget+ans_budget+np.sum(np.array(individual_r), axis=0).tolist())

                    episodes.append(episode)
                    episode_counts += 1

            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            self.buffer.store_episode(episode_batch)
            if self.buffer.current_size * self.env._max_steps > self.args.warm_up_steps:
                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    self.agents.train(mini_batch, train_steps, episode_counts)
                    train_steps += 1
        episode_reward = self.evaluate()
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + " >> {}/{} episodes , avg evaluate score : {:.3f}, agent_budget : {}"
            .format(episode_counts, self.args.max_episodes, episode_reward, ask_budget))
        self.episode_rewards.append(episode_reward)
        self.plt(num)

    def evaluate(self):
        episode_rewards = 0
        episodes = self.args.test_episode if (self.args.load_model and self.args.evaluate) else self.args.evaluate_episode
        for ep in range(episodes):
            episode, episode_reward, _, step, _, _, _, _ = self.rolloutWorker.generate_episode(evaluate=True)
            if self.args.load_model and self.args.evaluate:
                a_r = np.sum(episode['r'], axis=1).squeeze(0)
                print('episode {} has finished. reward={},  agent_reward={}, environment_step={}\n'.format(ep+1, round(
                    episode_reward, 4), a_r, step))
            episode_rewards += episode_reward
        return episode_rewards / episodes

    def plt(self, num):
        plt.figure()
        plt.cla()

        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('episode*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')

        plt.savefig(self.args.plt_dir + 'plt.png', format='png')
        plt.close()
