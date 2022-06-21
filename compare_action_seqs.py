import numpy as np
method = "DECOMP_STATE_tj_tj_easy_fixed_autoencoder_action0.7"

no_comm_acts = np.load("saved_actions/actions_no_comm=True_" + method + ".npy",allow_pickle=True).item()
no_comm_isalives = np.load("saved_actions/isalive_no_comm=True_" + method + ".npy",allow_pickle=True).item()

comm_acts = np.load("saved_actions/actions_no_comm=False_" + method + ".npy",allow_pickle=True).item()
comm_isalives = np.load("saved_actions/isalive_no_comm=False_" + method + ".npy",allow_pickle=True).item()


all_same_actions = []
all_comm_goes = [] #count how many times the communicating agent goes at each step
all_actions = []
for seed in no_comm_acts.keys():
    no_comm_ep_acts = no_comm_acts.get(seed)
    no_comm_ep_isalive = no_comm_isalives.get(seed)

    comm_ep_acts = comm_acts.get(seed)
    comm_ep_isalive = comm_isalives.get(seed)

    found_instance = False


    ep_same_actions = [1]
    ep_comm_goes = [1]
    ep_all_actions = [1]

    #look for instances where both models took the exact same actions until there were multiple agents in the scene
    for no_comm_act, no_comm_isalive, comm_act, comm_isalive in zip(no_comm_ep_acts, no_comm_ep_isalive, comm_ep_acts, comm_ep_isalive):
        #if the agents took different actions before there were other agents in the environment
        if not np.array_equal(no_comm_act,comm_act) and not found_instance:
            break
        #if the agents took the same actions and there are no other agents in the environment
        if np.array_equal(no_comm_act,comm_act) and not found_instance and sum(no_comm_isalive) <= 1:
            continue

        if sum(no_comm_isalive) > 1 and sum(comm_isalive) > 1:
            found_instance = True

        no_comm_rel_i = [i for i,v in enumerate(no_comm_isalive) if v == 1]
        comm_rel_i = [i for i,v in enumerate(comm_isalive) if v == 1]

        rel_i = np.union1d(comm_rel_i, no_comm_rel_i).astype(int)

        if len(rel_i) == 0:
            continue

        no_comm_rel_act = np.take(no_comm_act, rel_i)
        comm_rel_act =  np.take(comm_act, rel_i)

        n_same = 0
        n_c_goes = 0
        for nca, ca in zip(no_comm_rel_act, comm_rel_act):
            if nca == ca:
                n_same += 1
            if ca == 0:
                n_c_goes += 1

        ep_same_actions.append(n_same)
        ep_comm_goes.append(n_c_goes)
        ep_all_actions.append(len(rel_i))

    all_same_actions.append(ep_same_actions)
    all_comm_goes.append(ep_comm_goes)
    all_actions.append(ep_all_actions)



res = np.zeros(40)
total_samples = np.zeros(40)
for ep_same_actions, ep_all_actions in zip(all_same_actions,all_actions):
    for i,t_same_action in enumerate(ep_same_actions):
        res[i] += t_same_action
        # res[i] /= 2
        # total_samples[i] += 1
        total_samples[i] += ep_all_actions[i]

res_comm_goes = np.zeros(40)

for ep_goes in all_comm_goes:
    for i,t_goes in enumerate(ep_goes):
        res_comm_goes[i] += t_goes
        # res_comm_goes[i] /= 2
        # total_samples_goes[i] += 1

print ("percentage of times comm and non coom agents took the same action")
print (res/total_samples)
print ("percentage of times comm agents took action 0")
print (res_comm_goes/total_samples)
print ("total number of episodes (not agents) used per additional step")
print (total_samples)
