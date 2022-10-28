import os, sys, subprocess

os.environ["OMP_NUM_THREADS"] = "1"
env = "minecraft"
seeds = [0]
# your models, graphs and tensorboard logs would be save in trained_models/{exp_name}
# methods = ['baseline_mac_easy_mha_autoencoder_contrastive1']
# methods = ['baseline_mac_easy_mha_compositional_contrastive1']
methods = ['baseline_mac_mha']
pretrain_exp_name = 'tj_easy_fixed_proto_autoencoder'
for method in methods:
    comp_beta = 0.1
    num_proto = 128
    num_epochs = 1000
    exp_name = "mc_" + method
    # discrete comm is true if you want to use learnable prototype based communication.
    discrete_comm = False
    if "proto" in method:
        discrete_comm = True
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
    variable_gate = False
    nprocesses = 1

    hid_size = 64
    lr = 0.0001
    # easy
    nagents = 3
    max_steps = 900
    # dim = 6
    epoch_size = 10

    comm_dim = hid_size
    # comm_action_zero = True

    run_str = f"python main.py --env_name {env} --nprocesses {nprocesses} --batch_size 100 --gamma .99 "+\
              f"--num_epochs {num_epochs} --epoch_size {epoch_size} --num_heads 1 "+\
              f"--gating_head_cost_factor {gating_head_cost_factor} --num_proto {num_proto} "+\
              f"--hid_size {hid_size} --comm_dim {comm_dim}  "+\
              f" --detach_gap 10 --lrate {lr} --comp_beta {comp_beta} "+\
              f"--save paper_models --load paper_models "+\
              f"--max_steps {max_steps} --nagents {nagents} "+\
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
    if 'mha' in method:
        run_str += '--mha_comm '
    if 'timmac' in method:
        run_str += '--timmac '
    elif 'mac' in method:
        run_str += '--mac --recurrent --rnn_type GRU '
    else:
        run_str += '--ic3net --recurrent '
    if 'preencode' in method:
        run_str += '--preencode '
    if 'vae' in method:
        run_str += '--vae '
    if 'vqvib' in method:
        run_str += '--use_vqvib '
    if 'compositional' in method:
        run_str += '--use_compositional '
    if 'contrastive' in method:
        run_str += '--contrastive '

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
        # with open("runLogs/" + exp_name + "Log.txt","wb") as out:
            # subprocess.Popen(run_str + f"--seed {seed}", shell=True, stdout=out)#, stderr=out)
        os.system(run_str + f"--seed {seed}")
    # sys.exit(0)
    # plot the avg and error graphs using multiple seeds.
    # os.system(f"python plot.py --env_name {env} --exp_name {exp_name} --nagents {nagents}")
