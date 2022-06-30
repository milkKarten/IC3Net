from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt


env = "traffic_junction"
seed = 1
num_heads = 1
soft_budget = 0.9
# your models, graphs and tensorboard logs would be save in trained_models/{exp_name}
method = "TEST_eta_comm_loss=1_tj_fixed_easy_learn_intent_gating_timmac_autoencoder_action_heads" + str(num_heads) + "_budget=" + str(soft_budget)

all_comms_to_loc = np.load("null_comm_eval_data/" + method + "_" + str(seed) + ".npy").item()
X = []
for proto, locs in all_comms_to_loc.items():
    print (proto)
    X.append(list(proto))

# db = DBSCAN(eps=1, min_samples=2).fit(X)
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
# labels = db.labels_
#
# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise_ = list(labels).count(-1)
#
# print("Estimated number of clusters: %d" % n_clusters_)
# print("Estimated number of noise points: %d" % n_noise_)
#
#
# # Black removed and is used for noise instead.
# unique_labels = set(labels)
# colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = [0, 0, 0, 1]
#
#     class_member_mask = labels == k
#
#     xy = X[class_member_mask & core_samples_mask]
#     plt.plot(
#         xy[:, 0],
#         xy[:, 1],
#         "o",
#         markerfacecolor=tuple(col),
#         markeredgecolor="k",
#         markersize=14,
#     )
#
#     xy = X[class_member_mask & ~core_samples_mask]
#     plt.plot(
#         xy[:, 0],
#         xy[:, 1],
#         "o",
#         markerfacecolor=tuple(col),
#         markeredgecolor="k",
#         markersize=6,
#     )
#
# plt.title("Estimated number of clusters: %d" % n_clusters_)
# plt.show()
