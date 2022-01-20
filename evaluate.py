import sys
import time
import signal
import argparse
import time, os
import numpy as np
import torch
import data
from models import *
from comm import CommNetMLP
from utils import *
from action_utils import parse_action_args
from evaluator import Evaluator
from args import get_args

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Given some data, runs 2D PCA on it and plots the results.
def plot_comms(_data, special=None, _pca=None, _ax=None):
    if _data.shape[1] > 2:
        if _pca is None:
            _pca = PCA(n_components=2)
            _pca.fit(_data)
        transformed = _pca.transform(_data)
    else:
        transformed = _data
    x = transformed[:, 0]
    y = transformed[:, 1]
    if _ax is None:
        fig, _ax = plt.subplots()
    pcm = _ax.scatter(x, y, s=20, marker='o', c='gray')
    if special is not None:
        special_transformed = _pca.transform(special) if _pca is not None else special
        _ax.scatter(special_transformed[:, 0], special_transformed[:, 1], s=30, c='red')
    return _pca


torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
torch.set_default_tensor_type('torch.DoubleTensor')


def load(path):
    # d = torch.load(path)
    # policy_net.load_state_dict(d['policy_net'])

    load_path = os.path.join(args.load, args.env_name, args.exp_name, "seed" + str(args.seed), "models")
    print(f"load directory is {load_path}")
    log_path = os.path.join(args.load, args.env_name, args.exp_name, "seed" + str(args.seed), "logs")
    print(f"log dir directory is {log_path}")
    save_path = load_path

    if 'model.pt' in os.listdir(load_path):
        print(load_path)
        model_path = os.path.join(load_path, "model.pt")

    else:
        all_models = sort([int(f.split('.pt')[0]) for f in os.listdir(load_path)])
        model_path = os.path.join(load_path, f"{all_models[-1]}.pt")

    d = torch.load(model_path)
    policy_net.load_state_dict(d['policy_net'])

parser = get_args()
init_args_for_env(parser)
args = parser.parse_args()

if args.ic3net:
    args.commnet = 1
    args.hard_attn = 1
    args.mean_ratio = 0

    # For TJ set comm action to 1 as specified in paper to showcase
    # importance of individual rewards even in cooperative games
    # if args.env_name == "traffic_junction":
    #     args.comm_action_one = True
# Enemy comm
args.nfriendly = args.nagents
if hasattr(args, 'enemy_comm') and args.enemy_comm:
    if hasattr(args, 'nenemies'):
        args.nagents += args.nenemies
    else:
        raise RuntimeError("Env. needs to pass argument 'nenemy'.")

env = data.init(args.env_name, args, False)

num_inputs = env.observation_dim
args.num_actions = env.num_actions

# Multi-action
if not isinstance(args.num_actions, (list, tuple)): # single action case
    args.num_actions = [args.num_actions]
args.dim_actions = env.dim_actions
args.num_inputs = num_inputs

# Hard attention
if args.hard_attn and args.commnet:
    # add comm_action as last dim in actions
    args.num_actions = [*args.num_actions, 2]
    args.dim_actions = env.dim_actions + 1

# Recurrence
if args.commnet and (args.recurrent or args.rnn_type == 'LSTM'):
    args.recurrent = True
    args.rnn_type = 'LSTM'


parse_action_args(args)

if args.seed == -1:
    args.seed = np.random.randint(0,10000)
torch.manual_seed(args.seed)

print(args)
print(args.seed)

if args.commnet:
    policy_net = CommNetMLP(args, num_inputs, train_mode=False)
elif args.random:
    policy_net = Random(args, num_inputs)

# this is what we are working with for IC3 Net predator prey.
elif args.recurrent:
    policy_net = RNN(args, num_inputs)
else:
    policy_net = MLP(args, num_inputs)

load(args.load)
policy_net.eval()
if not args.display:
    display_models([policy_net])

# share parameters among threads, but not gradients
for p in policy_net.parameters():
    p.data.share_memory_()

evaluator = Evaluator(args, policy_net, data.init(args.env_name, args))

st_time = time.time()

all_stats = []
all_comms_to_loc = {}
for i in range(500):
    ep, stat, all_comms, comms_to_loc = evaluator.run_episode()
    all_stats.append(stat)
    for k, v in comms_to_loc.items():
        np_k = k
        if all_comms_to_loc.get(np_k) is None:
            all_comms_to_loc[np_k] = {}
        matching_vals = all_comms_to_loc.get(np_k)
        for val in v:
            if val not in matching_vals.keys():
                matching_vals[val] = 0
            matching_vals[val] += 1
    print(stat)
print("All comms to loc", all_comms_to_loc)
total_episode_time = time.time() - st_time
average_stat = {}
for key in all_stats[0].keys():
    average_stat[key] = np.mean([stat.get(key) for stat in all_stats])
print("average stats is: ", average_stat)
print("time taken per step ", total_episode_time/stat['num_steps'])

protos_np = None
num_proto_cutoff = 9  # Or None if you want all of them.
try:
    all_comms_to_loc = {k: v for k, v in sorted(all_comms_to_loc.items(), key=lambda item: sum(item[1].values()))}
    # A bit gross, but first get proto network and then proto layer
    protos = policy_net.proto_layer.prototype_layer.prototypes
    # Pass the prototypes through sigmoid to get the actual values.
    constrained_protos = torch.sigmoid(protos)
    protos_np = constrained_protos.detach().cpu().numpy()
    protos_list = [proto for proto in all_comms_to_loc.keys()]
    if num_proto_cutoff is not None:
        protos_list = protos_list[:num_proto_cutoff]
    protos_np = np.asarray(protos_list)
    # print("Prototypes", protos_np)
except AttributeError:
    print("No prototypes in policy net, so not analyzing that.")
if protos_np is not None:
    pca_transform = plot_comms(protos_np)
    plt.close()

all_comms = np.array(all_comms)
num_agents = len(all_comms[0])
for i in range(num_agents):
    print(f"for agent{i} communication is: ", all_comms[:, i])

# Plot the locations associated with each prototype recorded during execution
proto_idx = 0
for proto, locs in all_comms_to_loc.items():
    grid = np.zeros((10, 10))
    for loc, count in locs.items():
        grid[loc[0], loc[1]] = count
    fig, ax = plt.subplots(1, 2)
    plot_comms(protos_np, np.expand_dims(np.asarray(proto), 0), pca_transform, ax[0])
    ax[1].imshow(grid, cmap='hot')
    plt.savefig("Proto" + str(proto_idx))
    # plt.show()
    plt.close()
    proto_idx += 1
    if num_proto_cutoff is not None and proto_idx >= num_proto_cutoff:
        break

def get_weighted_loc(_loc_dict):
    total_count = 0
    summed = np.zeros(2)
    for loc, count in _loc_dict.items():
        summed += count * np.asarray(loc)
        total_count += count
    return summed / total_count

# Lastly, compute a metric of correlation between distance in comm space and distance in grid.
proto_dists = []
space_dists = []
for proto1, locs1 in all_comms_to_loc.items():
    avg1 = get_weighted_loc(locs1)
    pca1 = pca_transform.transform(np.reshape(proto1, (1, -1)))
    print("PCA proto to location")
    print(proto1)
    print(pca1)
    print(avg1)
    for proto2, locs2 in all_comms_to_loc.items():
        if np.array_equal(proto1, proto2):
            continue
        proto_dist = np.linalg.norm(np.asarray(proto1) - np.asarray(proto2)) / (np.sqrt(args.comm_dim))
        avg2 = get_weighted_loc(locs2)
        space_dist = np.linalg.norm(avg1 - avg2) / (9 * np.sqrt(2))
        proto_dists.append(proto_dist)
        space_dists.append(space_dist)

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(np.asarray(proto_dists).reshape(-1, 1), np.asarray(space_dists).reshape(-1, 1))
plt.scatter(proto_dists, space_dists)
m = reg.coef_[0]
b = reg.intercept_[0]
print("M", m, "b", b)
plt.plot(proto_dists, m*np.asarray(proto_dists) + b)
plt.ylabel("Normalized L2 distance between physical locations")
plt.xlabel("Normalized L2 distance between prototype vectors")
plt.savefig("Correlation.png")
# plt.show()
