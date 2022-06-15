import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np

method = "TEST_tj_tj_easy_fixed_autoencoder_action0.7"
prop = "autoencoder_loss"
name = "paper_models/traffic_junction/" + method


mean_met = np.zeros(200)
for i in range(1):

    seed = i
    fp = name + "/seed" + str(seed) + "/logs/"

    met = np.load(fp + prop + ".npy")
    for s_i,s in enumerate(met):
        mean_met[s_i] += s
plt.plot(np.arange(0,len(mean_met)), mean_met/10)
# plt.savefig(method + '_'+prop+'.png')

plt.show()
