import sys
import gym
import ic3net_envs
from env_wrappers import *
try:
    from smac.env import StarCraft2Env
except ImportError as e:
    pass  # starcraft experiments not enabled
from graph_env.env.full import SaturnHuman
from graph_env.utils import World


def init(env_name, args, final_init=True):
    if env_name == 'levers':
        env = gym.make('Levers-v0')
        env.multi_agent_init(args.total_agents, args.nagents)
        env = GymWrapper(env)
    elif env_name == 'number_pairs':
        env = gym.make('NumberPairs-v0')
        m = args.max_message
        env.multi_agent_init(args.nagents, m)
        env = GymWrapper(env)
    elif env_name == 'predator_prey':
        env = gym.make('PredatorPrey-v0')
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == 'traffic_junction':
        env = gym.make('TrafficJunction-v0')
        # if args.display:
        #     env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == 'starcraft':
        env = StarCraft2Env(map_name=args.map_name, seed=args.seed, reward_only_positive=True)
        env = StarcraftWrapper(env)
    elif env_name == 'minecraft':
        env = SaturnHuman(seed=args.seed)
        env = MinecraftWrapper(env)
    else:
        raise RuntimeError("wrong env name")

    return env
