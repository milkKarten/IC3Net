import os, sys, subprocess

os.environ["OMP_NUM_THREADS"] = "1"
env = "starcraft"
# seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
# seeds = [1, 2, 3, 4, 5]
seeds = [3]
# seeds = [20]
# your models, graphs and tensorboard logs would be save in trained_models/{exp_name}
methods = ['baseline_autoencoder_action_pos']
pretrain_exp_name = ''
# for soft_budget in [.5]:
if True:
    for method in methods:
        num_epochs = 1000
        num_proto = 100
        exp_name = "sc_" + method
        soft_budget = 0.7
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
            comm_action_one = True
        nprocesses = 0
        lr = 0.001
        nagents = 3
        max_steps = 60

        run_str = f"python evaluate_starcraft.py --env_name {env} --nprocesses {nprocesses} "+\
                  f"--num_epochs {num_epochs} --epoch_size 10 "+\
                  f"--gating_head_cost_factor {gating_head_cost_factor} "+\
                  f"--hid_size {hid_size} --comm_dim {hid_size} "+\
                  f" --detach_gap 10 --ic3net "+\
                  f"--recurrent --load SC2_models "+\
                  f"--max_steps {max_steps} --nagents {nagents} "+\
                  f"--exp_name {exp_name} --save_every {save_every} "

        if discrete_comm:
            run_str += f"--discrete_comm --use_proto --num_proto {num_proto} "
        if comm_action_one:
            run_str += f"--comm_action_one  "
        if comm_action_zero:
            run_str += f"--comm_action_zero "
        if "autoencoder" in method:
            run_str += "--autoencoder "
        if "action" in method:
            run_str += "--autoencoder_action "

        # Important: If you want to restore training just use the --restore tag
        # run for all seeds
        for seed in seeds:
            # log_path = os.path.join("trained_models", env, exp_name, "seed" + str(seed), "logs")
            log_path = os.path.join("SC2_models", env, exp_name, "seed" + str(seed), "logs")
            if os.path.exists(log_path):
                run_str += f"--restore  "
            # with open("runLogs/" + exp_name + "Log.txt","wb") as out:
            #     subprocess.Popen(run_str + f"--seed {seed}", shell=True, stdout=out)#, stderr=out)
            os.system(run_str + f"--seed {seed}")
