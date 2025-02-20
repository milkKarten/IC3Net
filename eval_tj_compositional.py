import os, sys, subprocess

os.environ["OMP_NUM_THREADS"] = "1"
env = "traffic_junction"
# seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
# seeds = [1, 2, 3]
# seeds = [777]
seeds = [2]
# seeds = [3,4]
# seeds = [0,1,2,3,4]
# your models, graphs and tensorboard logs would be save in trained_models/{exp_name}
# methods = ["easy_proto_soft_minComm_autoencoder", "easy_proto_soft_minComm_autoencoder_action"]
# methods = ["easy_proto_soft_minComm_autoencoder"]
# methods = ["easy_proto_autoencoder_minComm"]
# methods = ['hard_fixed', 'hard_fixed_autoencoder']
# methods = ['baseline_easy_ic3net_autoencoder_action_mha']
# methods = ['baseline_mac_easy_mha_comm_action_zero']
# methods = ['baseline_mac_easy_mha_compositional_500','baseline_mac_easy_mha_autoencoder_vqvib_500']
# methods = ['baseline_mac_easy_mha_autoencoder_vqvib_100_32_0.01']
methods = [
            # 'baseline_mac_easy_mha_compositional_tokenfix_100_0loss_good',
            # 'baseline_mac_easy_mha_compositional_tokenfix_100_0.001loss_good',
            # 'baseline_mac_easy_mha_compositional_tokenfix_100_0.01loss_good',
            # 'baseline_mac_easy_mha_compositional_tokenfix_100_0.1loss_good',
            # 'baseline_mac_easy_mha_compositional_tokenfix_100_0.01loss_good2',
            # 'baseline_mac_easy_mha_compositional_tokenfix_100_0.01loss_good8',
            # 'baseline_mac_easy_mha_compositional_tokenfix_100_0.01loss_good16',
            # 'baseline_mac_easy_mha_compositional_tokenfix_100_0.01loss_good',
            # 'baseline_mac_medium_mha_compositional_tokenfix_100_0.01loss_good112',
            # 'baseline_mac_medium_mha_compositional_tokenfix_100_0.01loss_good1120',
            # 'baseline_mac_easy_mha_compositional_100_0.1loss_qual32',
            # 'baseline_mac_easy_mha_compositional_100_0.01loss_qual32',
            # 'baseline_mac_easy_mha_compositional_100_0.001loss_qual32',
            # 'baseline_mac_easy_mha_compositional_100_0loss_qual32'
            'baseline_mac_easy_mha_autoencoder_vqvib'
]
# methods = ['baseline_mac_medium_mha_compositional_tokenfix_100_0loss',
#             'baseline_mac_medium_mha_compositional_tokenfix_100_0.001loss',
#             'baseline_mac_medium_mha_compositional_tokenfix_100_0.01loss',
#             'baseline_mac_medium_mha_compositional_tokenfix_100_0.1loss']
# methods = ['baseline_mac_easy_mha_autoencoder_comm003_32']
# methods = ['baseline_hard_timmac_autoencoder_action']
# methods = ['baseline_medium_ic3net_autoencoder_action']
# methods = ['baseline_hard_timmac_mha_autoencoder_action',
#             'baseline_hard_timmac_mha_autoencoder_action_preencode',
#             'baseline_hard_timmac_autoencoder_action',
#             'baseline_hard_ic3net_mha_autoencoder_action']
# methods = ['hard_fixed_proto', 'hard_fixed_proto_autoencoder']
# methods = ["easy_proto_soft_minComm_autoencoder_action"]
pretrain_exp_name = 'tj_easy_fixed_proto_autoencoder'
# pretrain_exp_name = 'tj_easy_fixed_proto_autoencoder_action'
# for soft_budget in [.5]:
# for method, num_proto in zip(methods, [1120]):
    # comp_beta = 0.01
num_proto = 32
for method, comp_beta in zip(methods, [0.1,0.01,0.001,0]):
    if 'action' in method:
        pretrain_exp_name = 'tj_easy_fixed_proto_autoencoder_action'
    if "easy" in method:
        # protos_list = [16, 32, 64]
        protos_list = [32]
        num_epochs = 1000
    elif 'medium' in method:
        # protos_list = [56, 28, 112]
        protos_list = [112] # use 1 layer of redundancy
        comms_list = [64]
        num_epochs = 2500
    elif 'hard' in method:
        # protos_list = [144, 72, 288]
        protos_list = [128*2] # single redundancy
        # comms_list = [64]
        num_epochs = 3000
    exp_name = "tj_" + method
    vision = 0
    soft_budget = 0.7
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
    if "rew_cur" in method:
        reward_curriculum = True
    variable_gate = False
    if "var" in method:
        variable_gate = True
    nprocesses = 1
    if "medium" in method:
        hid_size = 64

        lr = 0.001
        nagents = 10
        max_steps = 40
        dim = 14
        add_rate_min = 0.05
        add_rate_max = 0.2
        difficulty = 'medium'
        epoch_size = 100//nprocesses
    elif "hard" in method:
        hid_size = 64

        lr = 0.001
        nagents = 20
        max_steps = 80
        dim = 18
        add_rate_min = 0.02
        add_rate_max = 0.05
        difficulty = 'hard'
        epoch_size = 160//nprocesses
    else:

        hid_size = 32
        lr = 0.001
        # easy
        nagents = 5
        max_steps = 20
        dim = 6
        add_rate_min = 0.1
        # add_rate_min = 0.1
        add_rate_max = 0.3
        difficulty = 'easy'
        epoch_size = 10

    comm_dim = hid_size
    # comm_action_zero = True

    run_str = f"python evaluate_compositional.py --env_name {env} --nprocesses {nprocesses} --batch_size 100 --gamma 1 "+\
              f"--num_epochs {num_epochs} --epoch_size {epoch_size} --num_heads 1 "+\
              f"--gating_head_cost_factor {gating_head_cost_factor} --num_proto {num_proto} "+\
              f"--hid_size {hid_size} --comm_dim {comm_dim} --soft_budget {soft_budget} "+\
              f" --detach_gap 10 --lrate {lr} --vision {vision} --comp_beta {comp_beta} "+\
              f"--save paper_models --load paper_models "+\
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
        #     subprocess.Popen(run_str + f"--seed {seed}", shell=True, stdout=out)#, stderr=out)
        os.system(run_str + f"--seed {seed}")
    # sys.exit(0)
    # plot the avg and error graphs using multiple seeds.
    # os.system(f"python plot.py --env_name {env} --exp_name {exp_name} --nagents {nagents}")
