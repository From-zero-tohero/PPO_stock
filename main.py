# This is a sample Python script.
import sys
import torch
import numpy as np
import os
from stock_env import StockTradingEnv
from stock_env import get_gym_env_args
from agent import AgentPPO
from train import train_agent
from stock_env import build_env
from agent import ActorPPO
from train import get_episode_return_and_step
import matplotlib.pyplot as plt

def load_torch_file(model, _path):
    state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

class Arguments:
    def __init__(self, agent, env_func=None, env_args=None):
        self.env_func = env_func  # env = env_func(*env_args)
        self.env_args = env_args  # env = env_func(*env_args)

        self.env_num = self.env_args['env_num']  # env_num = 1. In vector env, env_num > 1.
        self.max_step = self.env_args['max_step']  # the max step of an episode
        self.env_name = self.env_args['env_name']  # the env name. Be used to set 'cwd'.
        self.state_dim = self.env_args['state_dim']  # vector dimension (feature number) of state
        self.action_dim = self.env_args['action_dim']  # vector dimension (feature number) of action
        self.if_discrete = self.env_args['if_discrete']  # discrete or continuous action space

        self.agent = agent  # DRL algorithm
        self.net_dim = 2 ** 7  # the middle layer dimension of Fully Connected Network
        self.batch_size = 2 ** 7  # num of transitions sampled from replay buffer.
        self.mid_layer_num = 1  # the middle layer number of Fully Connected Network
        self.if_off_policy = self.get_if_off_policy()  # agent is on-policy or off-policy
        self.if_use_old_traj = False  # save old data to splice and get a complete trajectory (for vector env)
        if self.if_off_policy:  # off-policy
            self.max_memo = 2 ** 21  # capacity of replay buffer
            self.target_step = 2 ** 10  # repeatedly update network to keep critic's loss small
            self.repeat_times = 2 ** 0  # collect target_step, then update network
        else:  # on-policy
            self.max_memo = 2 ** 12  # capacity of replay buffer
            self.target_step = self.max_memo  # repeatedly update network to keep critic's loss small
            self.repeat_times = 2 ** 4  # collect target_step, then update network

        '''Arguments for training'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.learning_rate = 2 ** -12  # 2 ** -15 ~= 3e-5
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3

        '''Arguments for device'''
        self.worker_num = 2  # rollout workers number pre GPU (adjust it to get high GPU usage)
        self.thread_num = 8  # cpu_num for pytorch, `torch.set_num_threads(self.num_threads)`
        self.random_seed = 0  # initialize random seed in self.init_before_training()
        self.learner_gpus = 0  # `int` means the ID of single GPU, -1 means CPU

        '''Arguments for evaluate'''
        self.cwd = None  # current working directory to save model. None means set automatically
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = +np.inf  # break training if 'total_step > break_step'

        '''Arguments for evaluate'''
        self.eval_gap = 2 ** 7  # evaluate the agent per eval_gap seconds
        self.eval_times = 2 ** 4  # number of times that get episode return

    def init_before_training(self):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.thread_num)
        torch.set_default_dtype(torch.float32)

        '''auto set cwd (current working directory)'''
        if self.cwd is None:
            self.cwd = f'./{self.env_name}_{self.agent.__name__[5:]}_{self.learner_gpus}'

        '''remove history'''
        if self.if_remove is None:
            self.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
        elif self.if_remove:
            import shutil
            shutil.rmtree(self.cwd, ignore_errors=True)
            print(f"| Arguments Remove cwd: {self.cwd}")
        else:
            print(f"| Arguments Keep cwd: {self.cwd}")
        os.makedirs(self.cwd, exist_ok=True)

    def get_if_off_policy(self):
        name = self.agent.__name__
        return all((name.find('PPO') == -1, name.find('A2C') == -1))  # if_off_policy


"""train and evaluate"""
def run():
    import sys
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    env = StockTradingEnv()
    env_func = StockTradingEnv
    env_args = get_gym_env_args(env=env, if_print=False)
    env_args['beg_idx'] = 0  # training set
    env_args['end_idx'] = 834  # training set

    args = Arguments(AgentPPO, env_func=env_func, env_args=env_args)
    args.target_step = args.max_step * 4
    args.reward_scale = 2 ** -7
    args.learning_rate = 2 ** -14
    args.break_step = int(5e5)

    args.learner_gpus = gpu_id
    args.random_seed += gpu_id + 1943

    # 训练
    train_agent(args)


def evaluate_models_in_directory(dir_path=None):
    if dir_path is None:
        gpu_id = "0"
        dir_path = f'StockTradingEnv-v2_PPO_{gpu_id}'
        print(f"| evaluate_models_in_directory: gpu_id {gpu_id}")
        print(f"| evaluate_models_in_directory: dir_path {dir_path}")
    else:
        gpu_id = -1
        print(f"| evaluate_models_in_directory: gpu_id {gpu_id}")
        print(f"| evaluate_models_in_directory: dir_path {dir_path}")

    model_names = [name for name in os.listdir(dir_path) if name[:6] == 'actor_']
    model_names.sort()

    env_func = StockTradingEnv
    env_args = {
        'env_num': 1,
        'env_name': 'StockTradingEnv-v2',
        'max_step': 1113,
        'state_dim': 151,
        'action_dim': 15,
        'if_discrete': False,

        'beg_idx': 834,  # testing set
        'end_idx': 1113,  # testing set
    }
    env = build_env(env_func=env_func, env_args=env_args)
    env.if_random_reset = False

    args = Arguments(AgentPPO, env_func=env_func, env_args=env_args)

    # device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    actor = ActorPPO(mid_dim=args.net_dim,
                     state_dim=args.state_dim,
                     action_dim=args.action_dim)

    for model_name in model_names:
        model_path = f"{dir_path}/{model_name}"
        load_torch_file(actor, model_path)

        # cumulative_returns_list = [get_episode_return_and_step(env, actor)[0] for _ in range(4)]
        cumulative_returns,episode_step,amount = get_episode_return_and_step(env, actor)

        print(f"cumulative_returns {cumulative_returns:9.3f}  {model_name}")

    plot(amount)


def plot(amount):
    # 画图
    plt.figure(figsize=(15, 6))
    plt.rcParams["font.size"] = 18

    plt.grid(visible=True, which="major", linestyle="-")
    plt.grid(visible=True, which="minor", linestyle="--", alpha=0.5)
    plt.minorticks_on()

    # plt.plot(epochs, train_acc, "o-", color="red", label="train_acc", linewidth=1, markersize=10)
    plt.plot(range(len(amount)), amount, color="blue", label="return", linewidth=2)

    plt.title("PPO Backtest")
    plt.xlabel("Date")
    plt.ylabel("return")

    plt.legend()
    plt.savefig("data/return.png")
    plt.close()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()
    # evaluate_models_in_directory()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
