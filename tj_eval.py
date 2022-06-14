import os, sys, subprocess

os.environ["OMP_NUM_THREADS"] = "1"

env = "traffic_junction"
seeds = [0]
# your models, graphs and tensorboard logs would be save in trained_models/{exp_name}
method = "easy_baseline_test_autoencoder_action"
# pretrain_exp_name = 'tj_EX_fixed_proto_comm_vs_protos_medium_p112_c64_d'
if "easy" in method:
    protos_list = [56]
    comms_list = [32]
    num_epochs = 2000
elif 'medium' in method:
    protos_list = [112] # use 1 layer of redundancy
    comms_list = [64]
    num_epochs = 3000
elif 'hard' in method:
    protos_list = [144, 72, 288]
    comms_list = [64]
    num_epochs = 4000
for num_proto in protos_list:
    for comm_dim in comms_list:
        exp_name = 'tj_' + method
        vision = 0
        # discrete comm is true if you want to use learnable prototype based communication.
        discrete_comm = False
        if "proto" in method:
            discrete_comm = True
        hid_size = 64
        save_every = 100
        # g=1. If this is set to true agents will communicate at every step.
        comm_action_one = False
        comm_action_zero = False
        # weight of the gating penalty. 0 means no penalty.
        gating_head_cost_factor = 0.
        if "fixed" in method or "baseline" in method:
            if not "var" in method:
                gating_head_cost_factor = 0
            comm_action_one = True
        nprocesses = 0
        lr = 0.001
        if "medium" in method:
            nagents = 10
            max_steps = 40
            dim = 14
            add_rate_min = 0.05
            add_rate_max = 0.2
            difficulty = 'medium'
        elif "hard" in method:
            nagents = 20
            max_steps = 80
            dim = 18
            add_rate_min = 0.02
            add_rate_max = 0.05
            difficulty = 'hard'
        elif "longer_easy" in method:
            nagents = 10
            max_steps = 30
            dim = 14
            add_rate_min = 0.1
            add_rate_max = 0.3
            difficulty = 'longer_easy'
        else:
            # easy
            nagents = 5
            max_steps = 20
            dim = 6
            add_rate_min = 0.1
            add_rate_max = 0.3
            difficulty = 'easy'


        run_str = f"python evaluate_starcraft.py --env_name {env} --nprocesses {nprocesses} "+\
                  f"--num_epochs {num_epochs} --epoch_size 10 "+\
                  f"--gating_head_cost_factor {gating_head_cost_factor} "+\
                  f"--hid_size {hid_size} --comm_dim {hid_size} "+\
                  f" --detach_gap 10 --lrate {lr} --ic3net --vision {vision} "+\
                  f"--recurrent --load paper_models "+\
                  f"--max_steps {max_steps} --dim {dim} --nagents {nagents} --add_rate_min {add_rate_min} --add_rate_max {add_rate_max} --curr_epochs 1000 --difficulty {difficulty} "+\
                  f"--exp_name {exp_name} --save_every {save_every} "

        if discrete_comm:
            run_str += f"--discrete_comm --use_proto --comm_dim {comm_dim} --num_proto {num_proto} "
        if comm_action_one:
            run_str += f"--comm_action_one  "
        if variable_gate:
            run_str += f"--variable_gate "
        if comm_action_zero:
            run_str += f"--comm_action_zero "
        if 'soft' in method:
            run_str += f"--load_pretrain --pretrain_exp_name {pretrain_exp_name} "

        if "minComm" in method:
            run_str += "--min_comm_loss --eta_comm_loss 1. "
        if "maxInfo" in method:
            run_str += "--max_info --eta_info 0.5 "
        if "autoencoder" in method:
            run_str += "--autoencoder "
        if "action" in method:
            run_str += "--autoencoder_action "
        if 'mha' in method:
            run_str += '--mha_comm '

        # Important: If you want to restore training just use the --restore tag
        # run for all seeds
        for seed in seeds:
            log_path = os.path.join("paper_models", env, exp_name, "seed" + str(seed), "logs")
            if os.path.exists(log_path):
                run_str += f"--restore  "
            # run_str += "> runLogs/" + exp_name + "Log.txt 2>&1 &"
            # cmd_args = run_str[:-1].split(" ")
            # print(cmd_args)
            # with open("runLogs/" + exp_name + "Log.txt","wb") as out:
            #     subprocess.Popen(run_str + f"--seed {seed}", shell=True, stdout=out)#, stderr=out)
            os.system(run_str + f"--seed {seed}")
