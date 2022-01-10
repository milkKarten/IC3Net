import os, sys, subprocess

os.environ["OMP_NUM_THREADS"] = "1"

# TODO: Run proto version
# explicity add to reward function to brake before intersection?
# does it make sense to have two way communication before action
# interleave a small amount of supervised data with some self-play
# send email to dana to setup box
# try keeping spawning rate constant
env = "traffic_junction"
# seeds = [1, 2]
seeds = [777]
# seeds = [20]
# your models, graphs and tensorboard logs would be save in trained_models/{exp_name}
# methods = ["fixed_proto_comm_vs_protos_junction_easy"]
# methods = ["medium_baseline_fixed_proto", "medium_baseline_fixed_continuous", "medium_baseline_G_proto", "medium_baseline_G_continuous"]
# methods = ["easy_baseline_fixed_proto", "easy_baseline_fixed_continuous", "easy_baseline_G_proto", "easy_baseline_G_continuous"]
methods = ["soft_easy_proto_rew_cur"]
# methods = ["soft_medium_proto_rew_cur"]
# methods = ["medium_fixed_continuous", "medium_G_proto", "medium_G_continuous"]
# run baseline with no reward on the gating function
# G - IC3net with learned gating function
# exp_name = "tj_g0.01_test"
# for reward_curr_start, reward_curr_end in zip([1500, 1250, 1800],[1900, 2000, 2000]):
pretrain_exp_name = 'tj_EX_fixed_proto_comm_vs_protos_easy_p56_c32'
# pretrain_exp_name = 'tj_EX_fixed_proto_comm_vs_protos_medium_p112_c64_d'
for soft_budget in [.10,.50,.90]:
    for method in methods:
        if "easy" in method:
            # protos_list = [14, 28, 56]
            protos_list = [56]
            # comms_list = [8, 32, 128]
            comms_list = [32]
            num_epochs = 2000
        elif 'medium' in method:
            # protos_list = [56, 28, 112]
            protos_list = [112] # use 1 layer of redundancy
            comms_list = [64]
            num_epochs = 3000
        elif 'hard' in method:
            protos_list = [144, 72, 288]
            comms_list = [64]
            num_epochs = 4000
        for num_proto in protos_list:
            for comm_dim in comms_list:
                exp_name = "tj_EX_" + method
                # exp_name = "tj_EX_" + method + "_p" + str(num_proto) + "_c" + str(comm_dim)
                vision = 0
                # discrete comm is true if you want to use learnable prototype based communication.
                discrete_comm = False
                if "proto" in method:
                    discrete_comm = True
                hid_size = 128
                save_every = 100
                # g=1. If this is set to true agents will communicate at every step.
                comm_action_one = False
                comm_action_zero = False
                # weight of the gating penalty. 0 means no penalty.
                # gating_head_cost_factor = rew
                gating_head_cost_factor = 0.1
                if "baseline" in method:
                    gating_head_cost_factor = 0
                if "fixed" in method:
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
                nprocesses = 16
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


                run_str = f"python main.py --env_name {env} --nprocesses {nprocesses} --display "+\
                          f"--num_epochs {num_epochs} --epoch_size 10 "+\
                          f"--gating_head_cost_factor {gating_head_cost_factor} "+\
                          f"--hid_size {hid_size} --soft_budget {soft_budget} "+\
                          f" --detach_gap 10 --lrate {lr} --ic3net --vision {vision} "+\
                          f"--recurrent "+\
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
                if reward_curriculum:
                    run_str += f"--gate_reward_curriculum "
                if 'junction' in method:
                    run_str += f"--use_tj_curric "
                if 'soft' in method:
                    run_str += f"--use_soft_constraint --load_pretrain --pretrain_exp_name {pretrain_exp_name} "

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
