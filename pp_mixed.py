import os

# specify environment name
env = "predator_prey"

# specify all the seeds you want to run the experiment on.
# seeds = [1, 2, 3]
seeds = [777]
methods = ["fixed_proto", "fixed_continuous", "G_proto", "G_continuous"]

for method in methods:
    # for predator-prey there are 3 modes: cooperative, competitive and mixed.
    mode = "mixed"
    # your models, graphs and tensorboard logs would be save in trained_models/{exp_name}
    exp_name = "pp_EX_" + mode + "_" + method

    # specify the number of predators.
    nagents = 5

    # number of epochs you wish to train on.
    num_epochs = 500

    # size of the hidden layer in LSTM
    hid_size = 128

    # dimension of the grid in predator-prey.
    dim = 10

    # max steps per episode.
    max_steps = 40

    # specify the vision for the agents. 0 means agents are blind.
    vision = 1

    # checkpoint models after every 100th epoch.
    save_every = 100

    # weight of the gating penalty. 0 means no penalty.
    gating_head_cost_factor = 0.0

    # discrete comm is true if you want to use learnable prototype based communication.
    discrete_comm = False
    if not 'continuous' in method:
        discrete_comm = True

    # specify the number of prototypes you wish to use.
    num_proto = 58

    # dimension of the communication vector.
    comm_dim = hid_size

    # boolean to specify using protos
    use_protos = False
    if "proto" in method:
        use_protos = True

    # whether prey can comunication or not.
    enemy_comm = True

    # g=1. If this is set to true agents will communicate at every step.
    comm_action_one = False

    nprocesses = 12

    if "fixed" in method:
        if not "var" in method:
            gating_head_cost_factor = 0
        comm_action_one = True
    # run for all seeds
    run_str = f"python main.py --env_name {env} --exp_name {exp_name} "+\
              f"--nagents {nagents} --mode {mode} "+\
              f"--nprocesses {nprocesses} --gating_head_cost_factor {gating_head_cost_factor} --num_epochs {num_epochs} "+\
              f"--hid_size {hid_size} --detach_gap 10 --lrate 0.001 "+\
              f"--recurrent "+\
              f"--dim {dim} --max_steps {max_steps} --ic3net --vision {vision} "+\
              f"--save_every {save_every} --comm_dim {comm_dim} "

    if discrete_comm:
        run_str += f"--discrete_comm "
    if use_protos:
        run_str += f"--use_proto --num_proto {num_proto} "
    if comm_action_one:
        run_str += f"--comm_action_one "
    if enemy_comm:
        run_str += f"--enemy_comm "

    for seed in seeds:
        log_path = os.path.join("trained_models", env, exp_name, "seed" + str(seed), "logs")
        if os.path.exists(log_path):
            run_str += f"--restore  "
        with open("runLogs/" + exp_name + "Log.txt","wb") as out:
            subprocess.Popen(run_str + f"--seed {seed}", shell=True, stdout=out)#, stderr=out)
        # os.system(run_str + f"--seed {seed}")
