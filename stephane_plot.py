import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np

method = "tj_tj_easy_fixed_autoencoder0.7"
prop = "success"



name = "paper_models/traffic_junction/" + method
seed = 0
fp = name + "/seed" + str(seed) + "/logs/"

succ = np.load(fp + prop + ".npy")
plt.plot(np.arange(0,len(succ)), succ)
# plt.savefig(method + '_'+prop+'.png')

plt.show()
