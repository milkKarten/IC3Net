import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np

# method = "DECOMP_STATE_tj_tj_hard_fixed_autoencoder_action0.7"
# prop = "success"
# name = "paper_models/traffic_junction/" + method



def get_runs_data(method,name,prop,color,n_runs=10):
    arr_size= 100
    mean_met = np.zeros(arr_size)
    all_met_res = [[] for i in range(arr_size)]
    for i in range(n_runs):

        seed = i
        fp = name + "/seed" + str(seed) + "/logs/"

        met = np.load(fp + prop + ".npy")
        for s_i,s in enumerate(met):
            if s_i >= arr_size:
                break
            mean_met[s_i] += s
            all_met_res[s_i].append(s)

    # all_met_res = [met_res_ if len(met_res_) >0 for met_res_ in all_met_res]
    all_met_res = np.array(all_met_res)
    print (all_met_res)
    # for i,res in enumerate()
    plt.plot(np.arange(0,len(mean_met)), np.mean(all_met_res,axis=1),color=color)
    plt.fill_between(np.arange(0,len(mean_met)), np.mean(all_met_res,axis=1) - np.std(all_met_res,axis=1),np.mean(all_met_res,axis=1) + np.std(all_met_res,axis=1),alpha=0.4,color=color)

    # plt.savefig(method + '_'+prop+'.png')

    # plt.show()
#
# prop = "fdm_loss"
# method_1 = "DECOMP_STATE_tj_tj_train_fdm_easy_fixed_autoencoder_action0.7"
# name_1 = "paper_models/traffic_junction/" + method_1
#
# get_runs_data(method_1,name_1,prop,"red",n_runs=3)
#
#
prop = "success"

#TEST_tj_learn_intent_gating_easy_fixed_autoencoder_action0 changes the eta_comm_loss to 0.1
#TEST_2_tj_learn_intent_gating_easy_fixed_autoencoder_action0 only uses the regularization loss


method_1 = "done_eta_comm_loss=1_0.03_tj_learn_intent_gating_easy_fixed_autoencoder_action0.7"
name_1 = "paper_models/traffic_junction/" + method_1
get_runs_data(method_1,name_1,prop,"orange",n_runs=10)

method_1 = "done_eta_comm_loss=1_0.03_tj_learn_intent_gating_easy_fixed_autoencoder_action0.5"
name_1 = "paper_models/traffic_junction/" + method_1
get_runs_data(method_1,name_1,prop,"green",n_runs=10)


method_1 = "done_eta_comm_loss=1_0.03_tj_learn_intent_gating_easy_fixed_autoencoder_action0.9"
name_1 = "paper_models/traffic_junction/" + method_1
get_runs_data(method_1,name_1,prop,"blue",n_runs=10)


# method_1 = "TEST_MLP_6_tj_train_fdm_easy_fixed_autoencoder_action0.7"
# name_1 = "paper_models/traffic_junction/" + method_1
#
# get_runs_data(method_1,name_1,prop,"red",n_runs=3)

# method_1 = "DECOMP_STATE_64_tj_comm_intent_1_easy_fixed_autoencoder_action0.7"
# name_1 = "paper_models/traffic_junction/" + method_1
#
# get_runs_data(method_1,name_1,prop,"blue",n_runs=3)

method_1 = "DECOMP_STATE_128_tj_comm_intent_1_easy_fixed_autoencoder_action0.7"
name_1 = "paper_models/traffic_junction/" + method_1

# get_runs_data(method_1,name_1,prop,"green",n_runs=10)

method_1 = "DECOMP_STATE_128_tj_comm_intent_2_easy_fixed_autoencoder_action0.7"
name_1 = "paper_models/traffic_junction/" + method_1

# get_runs_data(method_1,name_1,prop,"orange",n_runs=10)
#
#
# method_1 = "DECOMP_STATE_256_tj_comm_intent_1_easy_fixed_autoencoder_action0.7"
# name_1 = "paper_models/traffic_junction/" + method_1
#
# get_runs_data(method_1,name_1,prop,"purple",n_runs=3)

# prop = "autoencoder_loss"
method_1 = "DECOMP_STATE_tj_tj_easy_fixed_autoencoder_action0.7"
name_1 = "paper_models/traffic_junction/" + method_1
#
# method_2 = "AGG_STATE_tj_tj_easy_fixed_autoencoder_action0.7"
# name_2 = "paper_models/traffic_junction/" + method_2
#
# get_runs_data(method_1,name_1,prop,"red",n_runs=10)


method_1 = "DECOMP_STATE_128_tj_easy_fixed_autoencoder_action0.7"
name_1 = "paper_models/traffic_junction/" + method_1
#
# get_runs_data(method_1,name_1,prop,"blue",n_runs=10)
#
# get_runs_data(method_2,name_2,prop,"blue")
plt.show()
