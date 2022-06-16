import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np

method = "DECOMP_STATE_tj_tj_easy_fixed_autoencoder_action0.7"
prop = "autoencoder_loss"
name = "paper_models/traffic_junction/" + method


n_runs = 1
mean_met = np.zeros(200)
for i in range(n_runs):

    seed = i
    fp = name + "/seed" + str(seed) + "/logs/"

    met = np.load(fp + prop + ".npy")
    for s_i,s in enumerate(met):
        mean_met[s_i] += s
plt.plot(np.arange(0,len(mean_met)), mean_met/n_runs)
# plt.savefig(method + '_'+prop+'.png')

plt.show()
