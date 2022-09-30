import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = '16'
base_dir = '/Users/seth/Documents/research/IC3Net/paper_models/traffic_junction/'

fig, ax = plt.subplots(1)
# model_name = 'IMGS-MAC'
# model = 'baseline_autoencoder_action_pos'
# models = ['baseline_easy_timmac_mha_autoencoder_action_rewrite_heads1',
#             # 'baseline_easy_timmac_autoencoder_action_rewrite_heads1',
#             'baseline_easy_ic3net_mha_autoencoder_action_heads1',
#             # 'easy_fixed_autoencoder',
#             'easy_fixed']
models = ['baseline_better_medium_timmac_mha_autoencoder_action_lr001s1_skip0d_heads1',
            # 'baseline_better_medium_timmac_autoencoder_action_lr001s1_skip0d_heads1',
            # 'better_baseline_medium_ic3net_mha_autoencoder_action_heads1',
            'baseline_better_medium_ic3net_autoencoder_action_lr001s1_heads1',
            'baseline_better_medium_ic3net_heads1']
# model_names = ['TIM-MAC MHA',
#                 'TIM-MAC',
#                 'IMGS-MAC MHA',
#                 'IMGS-MAC',
#                 'IC3Net']
model_names = ['TIM-MAC',
                # 'TIM-MAC',
                'IMGS-MAC MHA',
                # 'IMGS-MAC',
                'IC3Net']
epochs = 200
# n = 16
for model, model_name in zip(models, model_names):
    model_data = []
    for seed in range(1,6):
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
        # data_success = np.average(data_success.reshape(-1, n), axis=1)
        while len(data_success) < epochs:
            data_success = np.append(data_success, data_success[-1])
        # epochs = max(len(data_success), epochs)
        data_success = data_success[:epochs]
        model_data.append(data_success)
    model_data = np.array(model_data)
    X = np.arange(epochs)

    import scipy.stats as st
    minInt, maxInt = st.t.interval(alpha=0.95, df=len(model_data)-1,
              loc=np.mean(model_data, axis=0),
              scale=st.sem(model_data))
    # print(minInt, maxInt)
    mu = model_data.mean(axis=0)
    # sigma = model_data.std(axis=0)
    ax.plot(X, mu, label=model_name)
    ax.fill_between(X, minInt, maxInt, alpha=0.5)
    # title = 'Easy Cts Traffic Junction'
    title = 'Medium Cts Traffic Junction'
ax.set_title(title+r' $\mu \pm \sigma$')
ax.legend(loc='lower right')
ax.set_xlabel('Epoch')
ax.set_ylabel('Success')
ax.grid()
# plt.savefig('/Users/seth/Documents/research/IC3Net/TIMMAC_figs/timmac_easytj',bbox_inches='tight')
plt.show()
