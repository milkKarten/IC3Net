import sys
import time
import signal
import argparse
import time, os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch
import data
from models import *
from comm import CommNetMLP
from utils import *
from action_utils import parse_action_args
from evaluator import Evaluator
from args import get_args
from mac import MAC

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
elif args.mac:
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
elif args.mac:
    policy_net = MAC(args, num_inputs)
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
print(policy_net.message_vocabulary)
all_stats = []
all_comms_to_loc0 = {}
all_comms_to_loc1 = {}
all_comms_to_loc2 = {}
for i in range(500):
    ep, stat, all_comms, comms_to_loc, comms_to_act, comms_to_full, comm_action_episode = evaluator.run_episode()
    all_stats.append(stat)
    for k, v in comms_to_full.items():
        np_k = k
        if all_comms_to_loc0.get(np_k) is None:
            all_comms_to_loc0[np_k] = {}
        matching_vals = all_comms_to_loc0.get(np_k)
        for val in v:
            if val not in matching_vals.keys():
                matching_vals[val] = 0
            matching_vals[val] += 1

# for action_level, all_comms_to_loc in enumerate([all_comms_to_loc0, all_comms_to_loc1, all_comms_to_loc2]):
for action_level, all_comms_to_loc in enumerate([all_comms_to_loc0]):
    all_comms = all_comms_to_loc
    # print("All comms to loc", all_comms_to_loc)
    print("action level", action_level)
    total_episode_time = time.time() - st_time
    average_stat = {}
    for key in all_stats[0].keys():
        average_stat[key] = np.mean([stat.get(key) for stat in all_stats])
    print("average stats is: ", average_stat)
    print("time taken per step ", total_episode_time/stat['num_steps'])

    protos_np = None
    num_proto_cutoff = None  # Or None if you want all of them.
    try:
        all_comms_to_loc = {k: v for k, v in sorted(all_comms_to_loc.items(), key=lambda item: sum(item[1].values()))}
        # A bit gross, but first get proto network and then proto layer
        # protos = policy_net.proto_layer.prototype_layer.prototypes
        # Pass the prototypes through sigmoid to get the actual values.
        # constrained_protos = torch.sigmoid(protos)
        # protos_np = constrained_protos.detach().cpu().numpy()
        protos_list = [proto for proto in all_comms_to_loc.keys()]
        # action_list = [proto[1] for proto in all_comms_to_loc.keys()]
        if num_proto_cutoff is not None:
            protos_list = protos_list[:num_proto_cutoff]
        protos_np = np.asarray(protos_list)
        print(protos_np[:10])
        print("Prototypes", protos_np.shape)
    except AttributeError:
        print("No prototypes in policy net, so not analyzing that.")
    if protos_np is not None:
        if policy_net.composition_dim == 2:
            pca_transform = protos_np
        else:
            pca_transform = plot_comms(protos_np)
            # plt.show()
            plt.close()

    # all_comms = np.array(all_comms)
    num_agents = args.nagents
    # for i in range(num_agents):
    #     print(f"for agent{i} communication is: ", all_comms[:, i])

    # Plot the locations associated with each prototype recorded during execution
    proto_idx = 0
    # print("testtesttest",all_comms_to_loc.items())
    # print("number of protos", len(all_comms_to_loc.items()))
    # print()
    # print()
    null_protos = []
    null_comms = 0
    total_comms = 0.
    null_str = ''
    active_protos = 0
    # for proto, locs in all_comms.items():
    #     # print(proto, locs)
    #     # grid = np.zeros((10, 10))
    #     grid = np.zeros((args.dim, args.dim))
    #     # print("locs items", locs.items())
    #     total_count = 0
    #     for loc, count in locs.items():
    #         # print("loc, count", loc, count)
    #         total_count += count
    #         grid[loc[0]-1, loc[1]-1] = count
    #     total_comms += total_count
    #     # print("locations:")
    #     grid_sum = grid.sum()
    #     num_locations = 0
    #     for loc, count in locs.items():
    #         percentage = grid[loc[0]-1, loc[1]-1] / grid_sum
    #         if percentage > .1:
    #             num_locations += 1
    #         # print(loc[0], loc[1], percentage)#, grid[loc[0], loc[1]], grid_sum)
    #     # print()
    #     active_protos += 1
    #     if policy_net.composition_dim == 2:
    #         fig, ax = plt.subplots(1, 2)
    #
    #         x = protos_np[:, 0]
    #         y = protos_np[:, 1]
    #         # print("protos np ", protos_np)
    #         ax[0].scatter(x, y, s=20, marker='o', c='gray')
    #         ax[0].scatter([proto[0]], [proto[1]], s=30, c='red')
    #         im = ax[1].imshow(grid, cmap='gray')
    #         plt.colorbar(im)
    #         plt.title(str(total_count))
    #         # plt.savefig(str(args.num_proto) + '/' + str(proto) + '.png')
    #         # plt.savefig(str(difficulty) + '/' + str(proto) + '.png')
    #         # plt.savefig(str(args.comp_beta) + '/' + str(proto) + '.png')
    #         # plt.savefig("tj_.5sparse_figs/Proto" + str(proto_idx) + str(action_level))
    #         # plt.savefig("/Users/seth/Documents/research/neurips/pca_easy_NON/protos_" + str(proto_idx) + str(action_level))
    #         # plt.show()
    #         plt.close()
    #     else:
    #         fig, ax = plt.subplots(1, 2)
    #         # print("protos np ", protos_np)
    #         plot_comms(protos_np, np.expand_dims(np.asarray(proto), 0), pca_transform, ax[0])
    #         im = ax[1].imshow(grid, cmap='gray')
    #         plt.colorbar(im)
    #         plt.title(str(total_count))
    #         # plt.savefig(str(args.comp_beta) + '/' + str(proto) + '.png')
    #         # plt.savefig("tj_.5sparse_figs/Proto" + str(proto_idx) + str(action_level))
    #         # plt.savefig("/Users/seth/Documents/research/neurips/pca_easy_NON/protos_" + str(proto_idx) + str(action_level))
    #         plt.show()
    #         # plt.close()
    #     proto_idx += 1
    #     if num_proto_cutoff is not None and proto_idx >= num_proto_cutoff:
    #         break
    #
    # look at location info
    all_locs = {}
    for proto, locs in all_comms.items():
        for loc, count in locs.items():
            tuple_loc = tuple(loc)
            if all_locs.get(tuple_loc) is None:
                all_locs[tuple_loc] = []
            all_locs[tuple_loc].append(tuple(proto))
    print(len(all_locs.items()))
    linearity_x = []
    linearity_y = []
    all_proto_locs_len = []
    for loc, protos in all_locs.items():
        fig, ax = plt.subplots(1, 2)
        protos_red = np.asarray(protos)
        all_proto_locs_len.append(len(protos_red))
        print(len(protos_red))
        xx = protos_red[:, 0]
        yy = protos_red[:, 1]
        # print("protos_red ", protos_red)
        x = protos_np[:, 0]
        y = protos_np[:, 1]
        ax[0].scatter(x, y, s=20, marker='o', c='gray')
        ax[0].scatter(xx, yy, s=30, c='red')
        grid = np.zeros((args.dim+1, args.dim+1))
        grid[loc[0], loc[1]] = 1
        im = ax[1].imshow(grid, cmap='gray')
        plt.colorbar(im)
        plt.title(str(protos))
        # plt.savefig(str(args.num_proto) + '/' + str(proto) + '.png')
        # plt.savefig(str(args.difficulty) + '_1120/' + str(loc) + '.png')
        # plt.savefig('A' + str(args.comp_beta) + '/' + str(protos) + '.png')
        # plt.savefig("tj_.5sparse_figs/Proto" + str(proto_idx) + str(action_level))
        # plt.savefig("/Users/seth/Documents/research/neurips/pca_easy_NON/protos_" + str(proto_idx) + str(action_level))
        plt.show()
        plt.close()
        # if loc[0] == 3 and not loc[1] == 3:
        #     linearity_x.append()
        #     linearity_x.append()
        # else:

    print('protos per loc', np.mean(all_proto_locs_len))
    print('proto length', evaluator.total_length / max(evaluator.total,1))

    # for proto in policy_net.message_vocabulary:
    #     print(proto, 'true')
    # with open("/Users/seth/Documents/research/neurips/nulls/"+args.exp_name+"/seed"+str(args.seed)+"/nulls.txt", 'w+') as f:
    #     f.write(null_str)
    # with open("/Users/seth/Documents/research/neurips/nulls/"+args.exp_name+"/seed"+str(args.seed)+"/percent_nulls.txt", 'w+') as f:
    # percent_nulls = "# null; total " + str(len(null_protos))+ "; "+ str(active_protos) + "\n % null comms" +  str(null_comms / total_comms) + '\n' + str(null_comms) + '/' + str(total_comms)
    #     f.write(percent_nulls)
    # print("% null comms: ", null_comms / total_comms)
