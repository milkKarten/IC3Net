import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = '16'
# base_dir = '/Users/seth/Documents/research/IC3Net/paper_models/traffic_junction/'
base_dir = '/home/milkkarten/research/IC3Net/paper_models/traffic_junction/'

fig, (ax, ax1) = plt.subplots(2,1)

models = [
        'baseline_mac_easy_mha_autoencoder_contrastive1'
        # 'easy_preencode_autoencoder'
]
model_names = ['success', r'$f^-$', r'$f^+$', 'complexity']
metrics = ['success', 'contrastive_loss_rand', 'contrastive_loss_future', 'autoencoder_loss']
mins = [0.65, 1., .99, 0.12]
maxs = [1.0, 1.35, 1.35, .15]
epochs = 1000
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# n = 16
linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
for model in models:
    for metric, _min, _max, linestyle, model_name in zip(metrics, mins, maxs, linestyles, model_names):
        model_data = []
        seed = 0
        s = str(seed)
        data_success = np.load(base_dir + 'tj_'+model+'/seed'+s+'/logs/'+metric+'.npy')
        if len(data_success) > epochs:
            data_success = data_success[:epochs]
        data_success = data_success[:epochs]
        data_success = np.convolve(data_success, np.ones((10,))/10, mode='valid')
        model_data = (np.array(data_success) - _min ) / (_max - _min)
        X = np.arange(model_data.shape[0])*10*100
        lbl = model_name
        ax.plot(X, model_data.reshape(-1), label=lbl, linestyle=linestyle)

model_names = ['success', 'autoencoder loss']
metrics = ['success', 'autoencoder_loss']
colors = ['black', 'cyan']
mins = [0.65, 0.01]
maxs = [1.0, .16]
epochs = 200
for model in ['easy_preencode_autoencoder']:
    for metric, _min, _max, linestyle, model_name, c in zip(metrics, mins, maxs, linestyles, model_names, colors):
        model_data = []
        seed = 6
        s = str(seed)
        data_success = np.load(base_dir + 'tj_'+model+'/seed'+s+'/logs/'+metric+'.npy')
        if len(data_success) > epochs:
            data_success = data_success[:epochs]
        data_success = data_success[:epochs]
        data_success = np.convolve(data_success, np.ones((5,))/5, mode='valid')
        model_data = (np.array(data_success) - _min ) / (_max - _min)
        X = np.arange(model_data.shape[0])*10*500
        lbl = model_name
        ax1.plot(X, model_data.reshape(-1), label=lbl, linestyle=linestyle, color=c)
    ax1.set_xlim(0,5e5)
    ax1.set_ylim(0,1)
    ax1.legend(loc='upper left')

title = 'Traffic Junction'
ax.set_xlim(0,5e5)
ax.set_ylim(0,1)
ax.set_title(title)
ax.legend(loc='center right', bbox_to_anchor=(1, 0.65))
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Normalized Value')
fig.set_size_inches(8, 6)
fig.text(0.5, 0.04, 'Steps', ha='center')
fig.text(0.02, 0.5, 'Normalized Value', va='center', rotation='vertical')
ax.grid()
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# plt.savefig('compositional_beta.png', bbox_inches='tight')
plt.savefig('easy_loss_analysis.png', bbox_inches='tight')
# plt.savefig('/Users/seth/Documents/research/IC3Net/TIMMAC_figs/timmac_easytj',bbox_inches='tight')
plt.show()
