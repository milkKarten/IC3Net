import numpy as np
import matplotlib.pyplot as plt

method0 = "tj_fixed_easy_timmac_autoencoder_action_heads1"
method1 = "eta_comm_loss=1_tj_fixed_easy_learn_intent_gating_timmac_autoencoder_action_heads1_budget=0.9"
method2 = "eta_comm_loss=1_tj_fixed_easy_learn_intent_gating_timmac_autoencoder_action_heads1_budget=0.7"
method3 = "eta_comm_loss=1_tj_fixed_easy_learn_intent_gating_timmac_autoencoder_action_heads1_budget=0.6"
method4 = "eta_comm_loss=1_tj_fixed_easy_learn_intent_gating_timmac_autoencoder_action_heads1_budget=0.5"
method5 = "eta_comm_loss=1_tj_fixed_easy_learn_intent_gating_timmac_autoencoder_action_heads1_budget=0.4"
method6 = "eta_comm_loss=1_tj_fixed_easy_learn_intent_gating_timmac_autoencoder_action_heads1_budget=0.3"

seed = 1
n_epochs = 500
# methods = [method0,method1,method2,method3,method4,method5,method6]
methods = [method1,method2,method3,method4,method5,method6]

# metric = "success"
metric = "percent_comm_0"

all_method_data = []
for method in methods:
    data = []
    arr = np.load("eval_logs/" + method + "_" + str(seed) +".npy",allow_pickle=True)
    for i in range(n_epochs):
        data.append(arr[i][metric])
    all_method_data.append(data)
    print (method)
    if metric == "success":
        print ("% success: " + str(np.mean(data)))
    else:
        print ("mean % comm: " + str(np.mean(data)/20))
    print ("\n")

# fig, ax = plt.subplots()
#
# ax.boxplot(all_method_data,positions=[1,0.9,0.7,0.6,0.5,0.4,0.3],widths=0.5)
# ax.set_xlim([0, 1.1])
#
# fig.patch.set_facecolor('white')
#
# plt.show()

# nagents = 5
# # methods = [method1,method2,method3,method4,method5]
# metric = "percent_comm_"


# all_method_data = []
# for method in methods:
#     data = []
#     for i in range(n_seeds):
#         arr = np.load("eval_logs/" + method + "_" + str(i) + ".npy",allow_pickle=True).item()
#         seed_data = []
#         for j in range (nagents):
#             seed_data.append(arr[metric + str(j)])
#         data.append(np.mean(seed_data))
#     all_method_data.append(data)

# fig, ax = plt.subplots()

# ax.boxplot(all_method_data,positions=[1,0.5,0.4,0.3],widths=0.05)
# ax.set_xlim([0, 1.1])
# fig.patch.set_facecolor('white')

plt.savefig("box_plot.png")
