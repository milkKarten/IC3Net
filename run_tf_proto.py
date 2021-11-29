import os

# TODO: Run proto version

env = "traffic_junction"
# seeds = [1, 2]
seeds = [777]

# your models, graphs and tensorboard logs would be save in trained_models/{exp_name}
methods = ["G", "fixed", "fixed_proto", "G_Proto"]
# G - IC3net with learned gating function
# exp_name = "tj_g0.01_test"
for method in methods:
    exp_name = "tj_" + method
    nagents = 5
    # discrete comm is true if you want to use learnable prototype based communication.
    discrete_comm = False
    if "proto" in method:
        discrete_comm = True
    num_epochs = 2000
    hid_size= 128
    dim = 6
    max_steps = 20
    vision = 0
    save_every = 100
    # g=1. If this is set to true agents will communicate at every step.
    comm_action_one = False
    # weight of the gating penalty. 0 means no penalty.
    gating_head_cost_factor = 0.01
    if "fixed" in method:
        gating_head_cost_factor = 0
        comm_action_one = True
    # specify the number of prototypes you wish to use.
    num_proto = 25
    # dimension of the communication vector.
    comm_dim = 16
    if not discrete_comm:
        comm_dim = hid_size

    run_str = f"python main.py --env_name {env} --nagents {nagents} --nprocesses 1 "+\
              f"--num_epochs {num_epochs} "+\
              f"--gating_head_cost_factor {gating_head_cost_factor} "+\
              f"--hid_size {hid_size} "+\
              f" --detach_gap 10 --lrate 0.001 --dim {dim} --max_steps {max_steps} --ic3net --vision {vision} "+\
              f"--recurrent "+\
              f"--add_rate_min 0.1 --add_rate_max 0.3 --curr_start 250 --curr_end 1250 --difficulty easy "+\
              f"--exp_name {exp_name} --save_every {save_every} "+\
              f"--use_proto --comm_dim {comm_dim} --num_proto {num_proto} " # may need to change this to not use prototypes

    if discrete_comm:
        run_str += f"--discrete_comm "
    if comm_action_one:
        run_str += f"--comm_action_one  "

    # Important: If you want to restore training just use the --restore tag
    # run for all seeds
    for seed in seeds:
        os.system(run_str + f"--seed {seed}")

    # plot the avg and error graphs using multiple seeds.
    os.system(f"python plot.py --env_name {env} --exp_name {exp_name} --nagents {nagents}")
