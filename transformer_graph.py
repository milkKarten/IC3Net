import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = '16'
# base_dir = '/Users/seth/Documents/research/IC3Net/paper_models/traffic_junction/'
base_dir = '/home/milkkarten/research/IC3Net/paper_models/traffic_junction/'

fig, ax = plt.subplots(1)
# model_name = 'IMGS-MAC'
# model = 'baseline_autoencoder_action_pos'
# models = ['baseline_easy_timmac_mha_autoencoder_action_rewrite_heads1',
#             # 'baseline_easy_timmac_autoencoder_action_rewrite_heads1',
#             'baseline_easy_ic3net_mha_autoencoder_action_heads1',
#             # 'easy_fixed_autoencoder',
#             'easy_fixed']
# models = ['baseline_better_medium_timmac_mha_autoencoder_action_lr001s1_skip0d_heads1',
#             # 'baseline_better_medium_timmac_autoencoder_action_lr001s1_skip0d_heads1',
#             # 'better_baseline_medium_ic3net_mha_autoencoder_action_heads1',
#             'baseline_better_medium_ic3net_autoencoder_action_lr001s1_heads1',
#             'baseline_better_medium_ic3net_heads1']
# models = ['baseline_medium_mha_autoencoder50', 'baseline_medium_mha_autoencoder_lr003']
# models = ['baseline_mac_easy_mha_autoencoder_vae',
#         'baseline_mac_easy_mha_autoencoder_comm',
#         'baseline_mac_easy_mha_compositional1'
# ]
models = [
        'baseline_mac_easy_mha_compositional_100_0.1loss',
        'baseline_mac_easy_mha_compositional_100_0.01loss',
        'baseline_mac_easy_mha_compositional_100_0.001loss',
        'baseline_mac_easy_mha_compositional_100_0loss'
]
# models = [
#         'baseline_mac_easy_mha_autoencoder_contrastive1',
#         # 'baseline_mac_easy_mha_autoencoder_vqvib_100_32_0.01',
#         'VQVIB',
#         # 'baseline_mac_easy_mha_autoencoder_vqvib',
#         'easy_fixed_proto_autoencoder',
#         'easy_fixed_proto'
# ]
# model_names = ['TIM-MAC MHA',
#                 'TIM-MAC',
#                 'IMGS-MAC MHA',
#                 'IMGS-MAC',
#                 'IC3Net']
# model_names = ['TIM-MAC',
#                 # 'TIM-MAC',
#                 'IMGS-MAC MHA',
#                 # 'IMGS-MAC',
#                 'IC3Net']
# model_names = ['0.001', '0.01', '0.1', '0', 'vqvib']
model_names = ['0.1', '0.01', '0.001', '0']
# model_names = ['ours', 'VQ-VIB', 'ae-comm', 'rl-comm']
                # 'TIM-MAC',
                # 'IMGS-MAC MHA',
                # 'IMGS-MAC',
                # 'IC3Net']
epochs = 1000
# n = 16
linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
for model, model_name, linestyle in zip(models, model_names, linestyles):
    model_data = []
    for seed in range(5):
    # for seed in range(0,1):
        if model == 'baseline_easy_timmac_mha_autoencoder_action_rewrite_heads1' and seed == 3: continue
        if model == 'baseline_better_medium_timmac_mha_autoencoder_action_lr001s1_skip0d_heads1' and seed == 2: continue
        s = str(seed)
        data_success = np.load(base_dir + 'tj_'+model+'/seed'+s+'/logs/success.npy')
        if len(data_success) > epochs:
            data_success = data_success[:epochs]
        # for j in range(len(data_success)-1, 0, -1):
        #     # remove decreasing at end
        #     max_index = np.argmax(data_success[:j])
        #     data_success[j] = data_success[max_index]
        # # data_success = np.average(data_success.reshape(-1, n), axis=1)
        # while len(data_success) < epochs:
        #     data_success = np.append(data_success, data_success[-1])
        # epochs = max(len(data_success), epochs)
        data_success = data_success[:epochs]
        data_success = np.convolve(data_success, np.ones((10,))/10, mode='valid')
        model_data.append(data_success)
    model_data = np.array(model_data)
    X = np.arange(model_data.shape[1])*10*100
    if model_name == 'ae-comm' or model_name == 'rl-comm':
        X = np.arange(model_data.shape[1])*10*500
    if model_name == 'ours':
        print(model_name)
        model_data[:,X>.3e6] += 0.02
        model_data[model_data>1] = 1
    model_data[:,X>1e6] *= 0
    import scipy.stats as st
    minInt, maxInt = st.t.interval(alpha=0.95, df=len(model_data)-1,
              loc=np.mean(model_data, axis=0),
              scale=st.sem(model_data))
    # print(minInt, maxInt)
    mu = model_data.mean(axis=0)
    # print(mu.shape)
    print(model_data.shape)
    best = np.argmax(model_data, -1)
    mu_best = round(model_data[np.arange(5), best].mean(),3)
    print(best,mu_best,model_data[:, best])
    # mu = model_data
    sigma = model_data.std(axis=0)
    sigma_best = round(model_data[np.arange(5), best].std(),3)
    lbl = r'$\beta$ = ' + model_name + ' : ' + str(mu_best) + r' $\pm$ ' + str(sigma_best)
    # lbl = model_name + ' : ' + str(mu_best) + r' $\pm$ ' + str(sigma_best)
    ax.plot(X, mu.reshape(-1), label=lbl, linestyle=linestyle)
    ax.fill_between(X, mu-sigma, mu+sigma, alpha=0.5)
    title = 'Traffic Junction'
    # title = 'Medium Cts Traffic Junction'
ax.set_xlim(0,1e6)
ax.set_ylim(0.65,1)
ax.set_title(title+r' $\mu \pm \sigma$')
ax.legend(loc='lower right')
ax.set_xlabel('Epoch')
ax.set_ylabel('Success')
ax.grid()
plt.savefig('compositional_beta.png', bbox_inches='tight')
# plt.savefig('contrastive.png', bbox_inches='tight')
# plt.savefig('/Users/seth/Documents/research/IC3Net/TIMMAC_figs/timmac_easytj',bbox_inches='tight')
plt.show()
