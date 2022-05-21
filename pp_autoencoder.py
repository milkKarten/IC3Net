import os, sys, subprocess

os.environ["OMP_NUM_THREADS"] = "1"
env = "predator_prey"
# seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
seeds = [1, 2, 3, 4, 5]
# seeds = [777]
# seeds = [20]
# your models, graphs and tensorboard logs would be save in trained_models/{exp_name}
methods = ['hard_baseline', 'hard_baseline_proto'] #baseline for PREDATOR PREY exps
pretrain_exp_name = 'tj_easy_fixed_proto_autoencoder'
# for soft_budget in [.5]:
if True:
    for method in methods:
        if 'action' in method:
            pretrain_exp_name = 'pp_easy_fixed_proto_autoencoder_action'
        if "easy" in method:
            protos_list = [56]
            num_epochs = 500
        elif 'medium' in method:
            protos_list = [112] # use 1 layer of redundancy
            comms_list = [64]
            num_epochs = 1000
        elif 'hard' in method:
            protos_list = [100]
            num_epochs = 2000
        for num_proto in protos_list:
            exp_name = "pp_" + method
            #VISION IS 1 FOR PREDATORY PREY
            vision = 1
            soft_budget = 0.7
            # for predator-prey there are 3 modes: cooperative, competitive and mixed.
            mode = "cooperative"
            # whether prey can comunication or not.
            enemy_comm = False
            # discrete comm is true if you want to use learnable prototype based communication.
            discrete_comm = False
            if "proto" in method:
                discrete_comm = True
            hid_size = 64
            save_every = 100
            comm_action_one = False
            comm_action_zero = False
            # weight of the gating penalty. 0 means no penalty.
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
                nagents = 5
                max_steps = 40
                dim = 10
            elif "hard" in method:
                nagents = 10
                max_steps = 80
                dim = 20
            else:
                # easy
                nagents = 3
                max_steps = 20
                dim = 5
                vision = 0


            run_str = f"python main.py --env_name {env} --nprocesses {nprocesses} "+\
                      f"--num_epochs {num_epochs} --epoch_size 10 "+\
                      f"--gating_head_cost_factor {gating_head_cost_factor} "+\
                      f"--hid_size {hid_size} --comm_dim {hid_size} --soft_budget {soft_budget} "+\
                      f" --detach_gap 10 --lrate {lr} --ic3net --vision {vision} "+\
                      f"--recurrent --save paper_models --load paper_models  --mode {mode} "+\
                      f"--max_steps {max_steps} --dim {dim} --nagents {nagents} "+\
                      f"--pretrain_exp_name {pretrain_exp_name} "+\
                      f"--exp_name {exp_name} --save_every {save_every} "

            if discrete_comm:
                run_str += f"--discrete_comm --use_proto --num_proto {num_proto} "
            if comm_action_one:
                run_str += f"--comm_action_one  "
            if comm_action_zero:
                run_str += f"--comm_action_zero "
            if 'soft' in method:
                run_str += f"--load_pretrain --pretrain_exp_name {pretrain_exp_name} "

            if "minComm" in method:
                run_str += "--min_comm_loss --eta_comm_loss 1. "
            if "autoencoder" in method:
                run_str += "--autoencoder "
            if enemy_comm:
                run_str += f"--enemy_comm "

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
