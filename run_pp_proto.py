import os

# specify environment name
env = "predator_prey"

# specify all the seeds you want to run the experiment on.
# seeds = [1, 2, 3]
seeds = [1]

# for predator-prey there are 3 modes: cooperative, competitive and mixed.
mode = "parent_child"

# your models, graphs and tensorboard logs would be save in trained_models/{exp_name}
exp_name = "one_hot"
# exp_name = "proto_fixed1"

# specify the number of predators.
nagents = 1

# number of epochs you wish to train on.
num_epochs = 6000

# size of the hidden layer in LSTM
hid_size = 81  # Was 128 for proto; 81 for one-hot

# dimension of the grid in predator-prey.
dim = 9

# max steps per episode.
max_steps = 20

# specify the vision for the agents. 0 means agents are blind.
vision = 1

# checkpoint models after every 100th epoch.
save_every = 100

# weight of the gating penalty. 0 means no penalty.
gating_head_cost_factor = 0.0

# discrete comm is true if you want to use learnable prototype based communication.
discrete_comm = True

# specify the number of prototypes you wish to use.
num_proto = 81

# dimension of the communication vector.
comm_dim = 81

# boolean to specify using protos
use_protos = False

# whether prey can comunication or not.
enemy_comm = False

# g=1. If this is set to true agents will communicate at every step.
comm_action_one = True

add_comm_noise = False

# Important: If you want to restore training just use the --restore tag

# run for all seeds
for seed in seeds:
    if discrete_comm:
        if use_protos:
            os.system(f"python main.py --comm_action_one --enemy_comm --env_name {env} --exp_name {exp_name} "
                      f"--nagents {nagents} --mode {mode} --seed {seed} "
                      f"--nprocesses 1 --gating_head_cost_factor {gating_head_cost_factor} --num_epochs {num_epochs} "
                      f"--hid_size {hid_size} --detach_gap 10 --lrate 0.001 "
                      f"--dim {dim} --max_steps {max_steps} --ic3net --recurrent --vision {vision} "
                      f"--save_every {save_every} --discrete_comm --use_proto --comm_dim {comm_dim} " +\
                      str("--add_comm_noise " if add_comm_noise else " ") +\
                      f"--num_proto {num_proto}")
        else:
            os.system(f"python main.py --comm_action_one --enemy_comm --env_name {env} --exp_name {exp_name} "
                      f"--nagents {nagents} --mode {mode} --seed {seed} "
                      f"--nprocesses 1 --gating_head_cost_factor {gating_head_cost_factor} --num_epochs {num_epochs} "
                      f"--hid_size {hid_size} --detach_gap 10 --lrate 0.001 "
                      f"--dim {dim} --max_steps {max_steps} --ic3net --recurrent --vision {vision} "
                      f"--save_every {save_every} --discrete_comm --comm_dim {comm_dim} " +\
                      str("--add_comm_noise " if add_comm_noise else " ")+\
                      f"--num_proto {num_proto}")
    else:
        os.system(
            f"python main.py --env_name {env} --exp_name {exp_name} --nagents {nagents} --mode {mode} --seed {seed} "
            f"--nprocesses 1 --gating_head_cost_factor {gating_head_cost_factor} --num_epochs {num_epochs} "
            f"--hid_size {hid_size} --detach_gap 10 --lrate 0.001 "
            f"--dim {dim} --max_steps {max_steps} --ic3net --recurrent --vision {vision} --recurrent --save_every {save_every} "
            f"--use_proto --comm_dim {comm_dim} " +\
            str("--add_comm_noise " if add_comm_noise else " ") +\
            f"--num_proto {num_proto}")

# plot the avg and error graphs using multiple seeds.
os.system(f"python plot.py --env_name {env} --exp_name {exp_name} --nagents {nagents}")
