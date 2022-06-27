import numpy as np
import matplotlib.pyplot as plt

method0 = "DECOMP_STATE_128_tj_easy_fixed_autoencoder_action0.7"
method1 = "done_eta_comm_loss=1_0.03_tj_learn_intent_gating_easy_fixed_autoencoder_action0.9"
method2 = "done_eta_comm_loss=1_0.03_tj_learn_intent_gating_easy_fixed_autoencoder_action0.7"
method3 = "done_eta_comm_loss=1_0.03_tj_learn_intent_gating_easy_fixed_autoencoder_action0.5"
method4 = "done_eta_comm_loss=1_0.003_tj_learn_intent_gating_easy_fixed_autoencoder_action0.3"
method5 = "done_eta_comm_loss=1_0.003_tj_learn_intent_gating_easy_fixed_autoencoder_action0.1"
n_seeds = 10
# methods = [method0,method1,method2,method3,method4,method5]
# metric = "success"
#
#
# all_method_data = []
# for method in methods:
#     data = []
#     for i in range(n_seeds):
#         arr = np.load("eval_logs/" + method + "_" + str(i) + ".npy",allow_pickle=True).item()
#         data.append(arr[metric])
#     all_method_data.append(data)
#
# fig, ax = plt.subplots()
#
# ax.boxplot(all_method_data,positions=[1,0.9,0.7,0.5,0.3,0.1],widths=0.05)
# ax.set_xlim([0, 1.1])
# fig.patch.set_facecolor('white')
#
# plt.show()

nagents = 5
methods = [method1,method2,method3,method4,method5]
metric = "percent_comm_"


all_method_data = []
for method in methods:
    data = []
    for i in range(n_seeds):
        arr = np.load("eval_logs/" + method + "_" + str(i) + ".npy",allow_pickle=True).item()
        seed_data = []
        for j in range (nagents):
            seed_data.append(arr[metric + str(j)])
        data.append(np.mean(seed_data))
    all_method_data.append(data)

fig, ax = plt.subplots()

ax.boxplot(all_method_data,positions=[0.9,0.7,0.5,0.3,0.1],widths=0.05)
ax.set_xlim([0, 1.1])
fig.patch.set_facecolor('white')

plt.show()
