import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    if value.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def get_common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=10, help='random seed')
    parser.add_argument('--env_name', type=str, default='ma_gym:FindTreasure-v0', help='env name')
    parser.add_argument('--max_episodes', type=int, default=50000, help='total episodes')
    parser.add_argument('--warm_up_steps', type=int, default=5000, help='warm up steps')
    parser.add_argument('--save_cycle', type=int, default=10000, help='model save cycle')
    parser.add_argument('--reuse_network', type=str2bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--individual_rewards', type=str2bool, default=True, help='whether to use individual rewards for training')
    parser.add_argument('--eps', type=float, default=1e-8, help='RMSprop eps')
    parser.add_argument('--alg', type=str, default='lsaa', help='the algorithm to train the agent')
    parser.add_argument('--n_episodes', type=int, default=1, help='the number of episodes before once training')
    parser.add_argument('--last_action', type=str2bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
    parser.add_argument('--evaluate_cycle', type=int, default=100, help='how often to evaluate the model')
    parser.add_argument('--evaluate_episode', type=int, default=10, help='number of the episode to evaluate the agent')
    parser.add_argument('--test_episode', type=int, default=10, help='number of the episode to test the trained model')
    parser.add_argument('--load_model', default=False, action='store_true', help='whether to load the pretrained model')
    parser.add_argument('--evaluate', default=False, action='store_true', help='whether to evaluate the model')
    parser.add_argument('--render', default=False, action='store_true', help='whether to render the environment')
    parser.add_argument('--cuda', type=str2bool, default=False, help='whether to use the GPU')
    parser.add_argument('--pkl_dir', default='model/FindTreasure/lsaa/', help='trained model directory')
    args = parser.parse_args()

    return args


def get_lsaa_args(args):
    args.rnn_hidden_dim = 64
    args.lr = 5e-4

    args.epsilon = 1
    args.min_epsilon = 0.05
    anneal_steps = 50000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    args.train_steps = 1

    args.batch_size = 32
    args.buffer_size = int(1e4)
    args.consult_budget = 10000
    args.respond_budget = 50000

    args.target_update_cycle = 200

    args.grad_norm_clip = 10

    args.start_advice = 5000

    args.lam = 0.6

    args.variable_a = 0.5
    args.beta = 0.2

    args.temperature = 0.3

    args.tau_pos = 3.0
    args.tau_neg = 1.0

    args.k_min = 2
    args.k_max = 4
    args.rho = 10

    args.n_min_per_state = 30
    args.n_max_per_state = 1000
    return args
