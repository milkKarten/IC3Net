import os, sys, subprocess

os.environ["OMP_NUM_THREADS"] = "1"
env = "traffic_junction"
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
# seeds = [1, 2, 3]
# seeds = [777]
# seeds = [20]
# your models, graphs and tensorboard logs would be save in trained_models/{exp_name}
# methods = ['hard_soft_minComm_autoencoder', 'easy_soft_minComm_autoencoder'. 'easy_proto_soft_minComm_autoencoder']
methods = ['easy_soft_minComm_autoencoder']
# pretrain_files = ['tj_hard_fixed_autoencoder', 'tj_easy_fixed_autoencoder', 'tj_easy_fixed_proto_autoencoder']
pretrain_files = ['tj_easy_fixed_autoencoder']
for soft_budget in [0.9, 0.5, 0.3, 0.1]:
    for method,pretrain_exp_name  in zip(methods, pretrain_files):
        if "easy" in method:
            # protos_list = [14, 28, 56]
            protos_list = [56]
            num_epochs = 50
        elif 'medium' in method:
            # protos_list = [56, 28, 112]
            protos_list = [112] # use 1 layer of redundancy
            num_epochs = 100
        elif 'hard' in method:
            # protos_list = [144, 72, 288]
            protos_list = [128*2] # single redundancy
            # comms_list = [64]
            num_epochs = 200
        for num_proto in protos_list:
            exp_name = "tj_" + method + str(soft_budget)
            # exp_name = "tj_EX_" + method + "_p" + str(num_proto) + "_c" + str(comm_dim)
            vision = 0
            soft_budget = 0.7
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
            # gating_head_cost_factor = rew
            gating_head_cost_factor = 0.
            if "fixed" in method or "baseline" in method:
                if not "var" in method:
                    gating_head_cost_factor = 0
                comm_action_one = True
            # use reward curriculum
            reward_curriculum = False
            if "rew_cur" in method:
                reward_curriculum = True
            variable_gate = False
            if "var" in method:
                variable_gate = True
            nprocesses = 16
            lr = 0.003
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
            else:
                # easy
                nagents = 5
                max_steps = 20
                dim = 6
                add_rate_min = 0.1
                add_rate_max = 0.3
                difficulty = 'easy'


            run_str = f"python main.py --env_name {env} --nprocesses {nprocesses} "+\
                      f"--num_epochs {num_epochs} --epoch_size 10 "+\
                      f"--gating_head_cost_factor {gating_head_cost_factor} "+\
                      f"--hid_size {hid_size} --comm_dim {hid_size} --soft_budget {soft_budget} "+\
                      f" --detach_gap 10 --lrate {lr} --ic3net --vision {vision} "+\
                      f"--recurrent --save paper_models --load paper_models "+\
                      f"--max_steps {max_steps} --dim {dim} --nagents {nagents} --add_rate_min {add_rate_min} --add_rate_max {add_rate_max} --curr_epochs 1000 --difficulty {difficulty} "+\
                      f"--exp_name {exp_name} --save_every {save_every} "

            if discrete_comm:
                run_str += f"--discrete_comm --use_proto --num_proto {num_proto} "
            if comm_action_one:
                run_str += f"--comm_action_one  "
            if variable_gate:
                run_str += f"--variable_gate "
            if comm_action_zero:
                run_str += f"--comm_action_zero "
            if reward_curriculum:
                run_str += f"--gate_reward_curriculum "
            if 'junction' in method:
                run_str += f"--use_tj_curric "
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

            # Important: If you want to restore training just use the --restore tag
            # run for all seeds
            for seed in seeds:
                # log_path = os.path.join("trained_models", env, exp_name, "seed" + str(seed), "logs")
                log_path = os.path.join("paper_models", env, exp_name, "seed" + str(seed), "logs")
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
