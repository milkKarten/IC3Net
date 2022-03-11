import os, sys, subprocess

os.environ["OMP_NUM_THREADS"] = "1"
env = "starcraft"
seeds = [777]
methods = ["baseline"]
pretrain_exp_name = ''
# for soft_budget in [.10,.50,.90]:
dir = '/Users/seth/Documents/research/TorchCraft'
for method in methods:
    num_epochs = 1000
    exp_name = "sc_" + method
    hid_size = 128
    comm_dim = hid_size
    num_proto = hid_size
    # discrete comm is true if you want to use learnable prototype based communication.
    discrete_comm = False
    if "proto" in method:
        discrete_comm = True
    save_every = 100
    comm_action_one = False
    comm_action_zero = False
    # weight of the gating penalty. 0 means no penalty.
    gating_head_cost_factor = 0.1
    if "fixed" in method or "baseline" in method:
        if not "var" in method:
            gating_head_cost_factor = 0
        comm_action_one = True
    if not discrete_comm:
        comm_dim = hid_size
    # use reward curriculum
    reward_curriculum = False
    if "rew_cur" in method:
        reward_curriculum = True
    variable_gate = False
    if "var" in method:
        variable_gate = True
    nprocesses = 0
    lr = 0.002
    soft_budget = 1.


    run_str = f"python main.py --env_name {env} --nprocesses {nprocesses} --display "+\
              f"--num_epochs {num_epochs} --epoch_size 10 "+\
              f"--gating_head_cost_factor {gating_head_cost_factor} "+\
              f"--hid_size {hid_size} --soft_budget {soft_budget} "+\
              f" --detach_gap 10 --lrate {lr} --ic3net "+\
              f"--recurrent --task_type explore --torchcraft_dir={dir} "+\
              "--recurrent --rnn_type LSTM --detach_gap 10 --stay_near_enemy --explore_vision 10 --step_size 16 "+\
              "--frame_skip 8 --nenemies 1 --our_unit_type 34 --enemy_unit_type 34 --init_range_end 150 "+\
              f"--max_steps 60 --nagents 10 "+\
              f"--exp_name {exp_name} --save_every {save_every} "

    if discrete_comm:
        run_str += f"--discrete_comm --use_proto --comm_dim {comm_dim} --num_proto {num_proto} "
    if comm_action_one:
        run_str += f"--comm_action_one  "
    if variable_gate:
        run_str += f"--variable_gate "
    if comm_action_zero:
        run_str += f"--comm_action_zero "
    if reward_curriculum:
        run_str += f"--gate_reward_curriculum "
    if 'soft' in method:
        run_str += f"--load_pretrain --pretrain_exp_name {pretrain_exp_name} "

    # Important: If you want to restore training just use the --restore tag
    # run for all seeds
    for seed in seeds:
        log_path = os.path.join("trained_models", env, exp_name, "seed" + str(seed), "logs")
        if os.path.exists(log_path):
            run_str += f"--restore  "
        # run_str += "> runLogs/" + exp_name + "Log.txt 2>&1 &"
        # cmd_args = run_str[:-1].split(" ")
        # print(cmd_args)
        with open("runLogs/" + exp_name + "Log.txt","wb") as out:
            subprocess.Popen(run_str + f"--seed {seed}", shell=True, stdout=out)#, stderr=out)
        # os.system(run_str + f"--seed {seed}")
    # sys.exit(0)
    # plot the avg and error graphs using multiple seeds.
    # os.system(f"python plot.py --env_name {env} --exp_name {exp_name} --nagents {nagents}")
