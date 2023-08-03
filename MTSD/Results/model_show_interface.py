import numpy as np
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt

'''your own path of all the results of models'''
path = '/Users/yawenzheng/Desktop/paper/MTS-DGCHN-master/MTSD/data_from_models'

names = ['ESTCNN_76', 'ECNN_76',
         'ECNN_BASE_CM_76', 'ECNN_BASE_PCM_76',
         'ECNN_BASE_CM_SA_76',  'ECNN_BASE_PCM_SA_76',
         'MTSD_76']
labels = ['ESTCNN', 'ECNN',
          'ECNN-Base + CM',
          'ECNN-Base + PCM',
          'ECNN-Base + CM + SA',
          'ECNN-Base + PCM + SA',
          'MTSD (ECNN-Base + PCM + SA + EI)']

fig_3 = plt.figure(figsize=(9, 9), num=3)
ax1_0_3 = plt.subplot(221)
ax1_1_3 = plt.subplot(212)
ax1_0_3.spines['top'].set_visible(False)
ax1_0_3.spines['right'].set_visible(False)
ax1_1_3.set_xlim(0, 50)
ax1_1_3.set_ylim(50, 91.5)
ax1_0_3.set_xlim(40, 50)
ax1_0_3.set_ylim(84, 91.5)
ax1_1_3.tick_params(axis='x', labelsize=11)
ax1_1_3.tick_params(axis='y', labelsize=12)
ax1_0_3.tick_params(axis='x', labelsize=11)
ax1_0_3.tick_params(axis='y', labelsize=12)
con1_3 = ConnectionPatch(xyA=(40, 84), coordsA=ax1_0_3.transData,
                       xyB=(40, 91.5), coordsB=ax1_1_3.transData, color='#707070')
con2_3 = ConnectionPatch(xyA=(50, 84), coordsA=ax1_0_3.transData,
                       xyB=(50, 91.5), coordsB=ax1_1_3.transData, color='#707070')
fig_3.add_artist(con1_3)
fig_3.add_artist(con2_3)
ax1_1_3.set_xlabel('Epochs', fontdict={'fontsize':13, 'fontname':'Arial'}, labelpad=8.5)
ax1_1_3.set_ylabel('Accuracy', fontdict={'fontsize':14, 'fontname':'Arial'}, labelpad=10.5)

color_ = ['#005795', '#56a8da',
          '#024a2b', '#059c68',
          '#923f0f', '#e77d3b',
          '#769a1f', '#a5ce5e']

for j, name in enumerate(names):
    mean_te_a = np.load(path + name + '_acc.npy')
    ax1_0_3.plot(mean_te_a, label=labels[j], color=color_[j])
    ax1_1_3.plot(mean_te_a, label=labels[j], color=color_[j])
    ax1_1_3.fill_between((40, 50), 50, 100, facecolor='#eeeeee', alpha=0.2)
    ax1_1_3.legend(loc='upper left', ncol=2)

# plt.show()
plt.savefig('/Users/yawenzheng/Desktop/paper/MTS-DGCHN-master/MTSD/data_from_models/result.jpg', dpi=450)