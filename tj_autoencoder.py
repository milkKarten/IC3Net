import os, sys, subprocess

os.environ["OMP_NUM_THREADS"] = "1"
env = "traffic_junction"
# seeds = [4]
seeds = [1,2,3]

methods = ['var_enc_fixed_easy_timmac_autoencoder']
# methods = ['fixed_easy_timmac_autoencoder_action']
pre_trained_intent_network_fp = "tj_fixed_easy_timmac_autoencoder_action_heads1"
past_comm_model_fp = "tj_fixed_easy_timmac_autoencoder_heads1"
# methods = ['hard_fixed_proto', 'hard_fixed_proto_autoencoder']
# methods = ["easy_proto_soft_minComm_autoencoder_action"]
pretrain_exp_name = 'tj_fixed_easy_timmac_autoencoder_heads1'
feat_net_path = "tj_learn_feat_net_fixed_easy_timmac_autoencoder_heads1_budget=1"
# pretrain_exp_name = 'tj_easy_fixed_proto_autoencoder_action'
# for soft_budget in [.5]:
# for soft_budget in [0.9,0.7,0.6,0.5,0.4,0.3,0.2,0.1]:


#TEST_BUG_tj_fixed_easy_timmac_autoencoder_action_heads1 - uses self attention + pe
#TEST_EXTRAFCNET_tj_fixed_easy_timmac_autoencoder_action_heads1 - uses self attention + pe + extra linear layer before VAE
#TEST_kld_weight_=0.1_tj_fixed_easy_timmac_autoencoder_action_heads1 - uses self attention + pe + VAE
#TEST_tj_fixed_easy_timmac_autoencoder_action_heads1 - uses self attention + pe + VAE
#TEST_NO_ATTENTION_tj_fixed_easy_timmac_autoencoder_action_heads1 - only linear layer + VAE


for soft_budget in [1]:
    num_heads = 1
    for method in methods:
        if 'action' in method:
            pretrain_exp_name = 'tj_fixed_easy_timmac_autoencoder_action_heads1'
        if "easy" in method:
            # protos_list = [14, 28, 56]
            protos_list = [56]
            num_epochs = 1200
        elif 'medium' in method:
            # protos_list = [56, 28, 112]
            protos_list = [112] # use 1 layer of redundancy
            comms_list = [64]
            num_epochs = 5000
        elif 'hard' in method:
            # protos_list = [144, 72, 288]
            protos_list = [128*2] # single redundancy
            # comms_list = [64]
            num_epochs = 10
        for num_proto in protos_list:
            
            exp_name = "NORMAL_VQ_VAE_tj_" + method + '_heads' + str(num_heads) + "_budget=" + str(soft_budget)
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

            if "easy" in method:
                nprocesses = 0
            else:
                nprocesses = 0
            if soft_budget < 1:
                lr = 0.03
            elif "var_enc" in method:
                lr=0.001
            else:
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


            run_str = f"python main.py --env_name {env} --nprocesses {nprocesses} --batch_size 500 "+\
                      f"--num_epochs {num_epochs} --epoch_size 10 --num_heads {num_heads} "+\
                      f"--gating_head_cost_factor {gating_head_cost_factor} "+\
                      f"--hid_size {hid_size} --comm_dim {hid_size} --soft_budget {soft_budget} "+\
                      f" --detach_gap 10 --lrate {lr} --vision {vision} "+\
                      f"--save paper_models --load paper_models "+\
                      f"--max_steps {max_steps} --dim {dim} --nagents {nagents} --add_rate_min {add_rate_min} --add_rate_max {add_rate_max} --curr_epochs 1000 --difficulty {difficulty} "+\
                      f"--exp_name {exp_name} --save_every {save_every} --gamma 1. "

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
            if "learn_intent_gating" in method:
                run_str += f"--learn_intent_gating "
                run_str += f"--intent_model_path {pre_trained_intent_network_fp} "
                run_str += "--min_comm_loss --eta_comm_loss 1. "
            if "var_enc" in method:
                run_str += f"--variational_enc "
            if "learn_feat_net" in method:
                run_str += f"--learn_feat_net "
                run_str += f"--load_pretrain --pretrain_exp_name {pretrain_exp_name} "
            if "load_feat_net" in method: 
                run_str += f"--load_feat_net "
                run_str += f"--feat_net_path {feat_net_path} "
            if "learn_past_comms" in method:
                run_str += f"--learn_past_comms "
                run_str += f"--past_comm_model_fp {past_comm_model_fp} "

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
            else:
                run_str += '--ic3net --recurrent '
            if 'preencode' in method:
                run_str += '--preencode '

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
