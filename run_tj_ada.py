import os, sys, subprocess

os.environ["OMP_NUM_THREADS"] = "1" # push this to repo

# TODO: Run proto version
# explicity add to reward function to brake before intersection?
# does it make sense to have two way communication before action
# interleave a small amount of supervised data with some self-play
# send email to dana to setup box
# try keeping spawning rate constant
env = "traffic_junction"
# seeds = [1, 2]
# seeds = [777]
seeds = [20]
# your models, graphs and tensorboard logs would be save in trained_models/{exp_name}
# methods = ["fixed"]
# methods = sys.argv[1:]
# print(methods)
# methods = ["fixed_proto", "G_Proto", "G", "fixed", "G_proto_bigproto",
#             "fixed_proto_bigproto", "G_proto_bigproto_bigcomm", "fixed_proto_bigproto_bigcomm"]
# methods = ["G_proto_bigproto_bigcomm", "G_proto_bigcomm"]
#methods = ["G_Proto", "G", "G_proto_bigproto_bigcomm"]
methods = ["G_proto_var", "G_proto_bigproto_bigcomm_var", "G_proto_bigproto_bigcomm32"]
# run baseline with no reward on the gating function
# G - IC3net with learned gating function
# exp_name = "tj_g0.01_test"
# for reward_curr_start, reward_curr_end in zip([1500, 1250, 1800],[1900, 2000, 2000]):
#for rew in [-.05, -.1, -.5]:
for rew in [-.1]:
    for method in methods:
        #exp_name = "tj_" + method + "_NEG_" + str(reward_curr_start) + "_" + str(reward_curr_end)
        exp_name = "tj_" + method + "_ADA_tGate3_" + str(rew)
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
        gating_head_cost_factor = rew
        if "baseline" in method:
            gating_head_cost_factor = 0
        if "fixed" in method:
            gating_head_cost_factor = 0
            comm_action_one = True
        # specify the number of prototypes you wish to use.
        num_proto = 25  # try to increase prototypes
        if "bigproto" in method:
            num_proto = 50  # try to increase prototypes
        # dimension of the communication vector.
        comm_dim = 16
        if "bigcomm" in method:
            comm_dim = 32
        if not discrete_comm:
            comm_dim = hid_size
        # use reward curriculum
        reward_curriculum = False
        gate_reward_max = -0.01
        gate_reward_min = 0.01
        if reward_curriculum:
            gating_head_cost_factor = gate_reward_min
        # reward_curr_start = 1500
        # reward_curr_end = 1900
        if "var" in method:
            variable_gate = True
        variable_gate_start = 500
        nprocesses = 16
        if "32" in method:
            nprocesses = 32
        run_str = f"python main.py --env_name {env} --nagents {nagents} --nprocesses {nprocesses} "+\
                  f"--num_epochs {num_epochs} "+\
                  f"--gating_head_cost_factor {gating_head_cost_factor} "+\
                  f"--hid_size {hid_size} "+\
                  f" --detach_gap 10 --lrate 0.001 --dim {dim} --max_steps {max_steps} --ic3net --vision {vision} "+\
                  f"--recurrent --optim_name Adadelta "+\
                  f"--add_rate_min 0.1 --add_rate_max 0.3 --curr_start 250 --curr_end 1250 --difficulty easy "+\
                  f"--exp_name {exp_name} --save_every {save_every} "+\
                  f"--use_proto --comm_dim {comm_dim} --num_proto {num_proto} " # may need to change this to not use prototypes

        if discrete_comm:
            run_str += f"--discrete_comm "
        if comm_action_one:
            run_str += f"--comm_action_one  "
        if variable_gate:
            run_str += f"--variable_gate --variable_gate_start {variable_gate_start} "
        if reward_curriculum:
            run_str += f"--gate_reward_curriculum --gate_reward_max {gate_reward_max} --gate_reward_min {gate_reward_min} "+\
                        f"--reward_curr_start {reward_curr_start} --reward_curr_end {reward_curr_end} "

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
                # subprocess.Popen(run_str, stdout=out)
                subprocess.Popen(run_str + f"--seed {seed}", shell=True, stdout=out)#, stderr=out)
                # subprocess.Popen(run_str[:-1].split(" "), stdout=out, stderr=out)
            # os.system(run_str + f"--seed {seed}")
        # sys.exit(0)
        # plot the avg and error graphs using multiple seeds.
        # os.system(f"python plot.py --env_name {env} --exp_name {exp_name} --nagents {nagents}")
