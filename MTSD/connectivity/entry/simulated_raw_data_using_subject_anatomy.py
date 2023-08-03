"""
=======================================
Simulate raw data using subject anatomy
=======================================

This example illustrates how to generate source estimates and simulate raw data
using subject anatomy with the :class:`mne.simulation.SourceSimulator` class.
Once the raw data is simulated, generated source estimates are reconstructed
using dynamic statistical parametric mapping (dSPM) inverse operator.
"""

# Author: Ivana Kojcic <ivana.kojcic@gmail.com>
#         Eric Larson <larson.eric.d@gmail.com>
#         Kostiantyn Maksymenko <kostiantyn.maksymenko@gmail.com>
#         Samuel Deslauriers-Gauthier <sam.deslauriers@gmail.com>

# License: BSD-3-Clause

# %%

import os.path as op

import numpy as np

import mne
from mne.datasets import sample
import matplotlib.pyplot as plt
# from Version_Main import LDL_0215
from connectivity.entry import LDL_0215
import pandas as pd

import os

print(__doc__)

# In this example, raw data will be simulated for the sample subject, so its
# information needs to be loaded. This step will download the data if it not
# already on your machine. Subjects directory is also set so it doesn't need
# to be given to functions.
data_path = sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
subject = 'sample'
meg_path = op.join(data_path, 'MEG', subject)

# First, we get an info structure from the sample subject.
fname_info = op.join(meg_path, 'sample_audvis_raw.fif')
info = mne.io.read_info(fname_info)
tstep = 1 / info['sfreq']
print(info)
print(info['description'])
print(info['acq_pars'])
# read_mont = mne.channels.find_layout(info)
#
# eeg_layout = mne.channels.make_eeg_layout(info)
# returns = eeg_layout.plot()
# plot_eeg_layout, eeg_pos, ax = returns[0], returns[1], returns[2]
# ax.scatter(eeg_pos['EEG 025'][0], eeg_pos['EEG 025'][1], s=5, c='green')

print('Done')

# read data

brain_temp = np.load('/Users/linya/Desktop/kde/for_rcm/0215/brain_temp.npy', allow_pickle=True)
vis_picks = np.load('/Users/linya/Desktop/kde/for_rcm/0215/vis_picks.npy', allow_pickle=True)
au_picks = np.load('/Users/linya/Desktop/kde/for_rcm/0215/au_picks.npy', allow_pickle=True)

ch_real_names = ['Fp1','Fpz','Fp2',
                 'AF7','AF3','AF4','AF8',
                 'F7','F5','F3','F1','Fz','F2','F4','F6','F8',
                 'FT7','FC5','FC3','FC1','FC2','FC4','FC6','FT8',
                 'TP7','T7','C5','C3','C1','Cz','C2','C4','C6','T8','TP8',
                 'P9','CP5','CP3','CP1','CP2','CP4','CP6','P10',
                 'P7','P5','P3','P1','Pz','P2','P4','P6','P8',
                  'PO7','PO3','PO4','PO8',
                 'O1','Oz','O2','lz']
print('len ch_real_names {}'.format(len(ch_real_names)))
au_eles = ['Fz', 'FC1', 'FC2', 'FC3', 'FC4', 'C4', 'C3', 'T7', 'T8', 'P3', 'P4', 'P7', 'P8', 'O1', 'O2', 'Oz']
au_eles = ['Fp1','Fpz','Fp2',
                 'F7','F5','F3','F1','Fz','F2','F4','F6','F8',
                 'FT7', 'FT8',
                 'TP7','T7', 'T8','TP8',
                 'P9','CP5','CP3','CP1','CP2','CP4','CP6','P10',
                 'P7','P5','P3','P1','Pz','P2','P4','P6','P8',
                  'PO7','PO3','PO4','PO8',
           ]

au_indexes = [ch_real_names.index(ele) for ele in au_eles]
vis_eles = ['Oz',
            'FT7','FT8',
                 'TP7','T7','T8','TP8',
                 'P9','CP5','CP3','CP1','CP2','CP4','CP6','P10',
                 'P7','P5','P3','P1','Pz','P2','P4','P6','P8',
                  'PO7','PO3','PO4','PO8',
                 'O1','O2',
            ]

# vis_eles = ['Oz',
#             'Fp1','Fpz','Fp2',
#                  'AF7','AF3','AF4','AF8',
#                  'F7','F5','F3','F1','Fz','F2','F4','F6','F8',
#                  'FT7','FC5','FC3','FC1','FC2','FC4','FC6','FT8',
#                  'TP7','T7','C5','C3','C1','Cz','C2','C4','C6','T8','TP8',
#                  'P9','CP5','CP3','CP1','CP2','CP4','CP6','P10',
#                  'P7','P5','P3','P1','Pz','P2','P4','P6','P8',
#                   'PO7','PO3','PO4','PO8',
#                  'O1','O2',
#             ]
vis_indexes = [ch_real_names.index(ele) for ele in vis_eles]
# au_indexes = vis_indexes
print(au_indexes)


save_path = os.path.join('/Users/linya/Desktop/research_files/thesis', 'topology')
if(os.path.exists(save_path)):
    pass
else:
    os.makedirs(save_path)

ave_dict = {}
loc_dict = {}
for i in range(18):
    if(i>=0):
        read_mont = mne.channels.find_layout(info)

        eeg_layout = mne.channels.make_eeg_layout(info)
        returns = eeg_layout.plot()
        plot_eeg_layout, eeg_pos, ax = returns[0], returns[1], returns[2]

        # remedied_data = np.load('/Users/linya/Desktop/kde/for_rcm/0215/' + str(i) + 'vis_r_final.npy')
        # original_data = np.load('/Users/linya/Desktop/kde/for_rcm/0215/' + str(i) + 'vis_o_final.npy')
        remedied_data = np.load('/Users/linya/Desktop/kde/for_rcm/0215/' + str(i) + '_r_final.npy')
        original_data = np.load('/Users/linya/Desktop/kde/for_rcm/0215/' + str(i) + '_o_final.npy')

        print('r\t', remedied_data.shape)
        re_cov_r = LDL_0215.My_Copula(remedied_data)
        print('r\t', re_cov_r)
        re_cov_o = LDL_0215.My_Copula(original_data)
        print('o\t', re_cov_o)

        train_data = re_cov_o
        data = pd.DataFrame(re_cov_r)
        cov_values = pd.DataFrame(data=data).corr().values
        remedied = cov_values
        print('cov_values: ', cov_values, '\t')
        print('cov_values: ', cov_values.shape, '\t')
        trans_train_data = pd.DataFrame(train_data)
        trans_cov_values = pd.DataFrame(data=trans_train_data).corr().values
        orginal = trans_cov_values
        print('original cov values: {}'.format(orginal.shape))

        # only get the upper triangle matrix
        rows, columns = remedied.shape
        remedied = np.insert(remedied, 52, 0, axis=1)
        remedied = np.insert(remedied, 52, 0, axis=0)
        orginal = np.insert(orginal, 52, 0, axis=1)
        orginal = np.insert(orginal, 52, 0, axis=0)
        print('changed {}'.format(remedied.shape))
        for row in range(rows+1):
            for col in range(columns+1):
                if(row == col or row == 52 or col == 52):
                    pass
                else:
                    # if(row == au_indexes[0] and col in au_indexes):
                    if(row in au_indexes and col in au_indexes):

                        if(len(ave_dict.keys())==0):
                            ave_dict[str(row)+','+str(col)]=[]
                        elif(str(row)+','+str(col) in list(ave_dict.keys())):
                            pass
                        else:
                            ave_dict[str(row)+','+str(col)]=[]

                        remedied_value = remedied[row, col]
                        original_value = orginal[row, col]
                        name_node_one = 'EEG 0' + str(row+1)
                        name_node_another = 'EEG 0' + str(col+1)
                        try:
                            pos_node_one = eeg_pos[name_node_one]
                        except Exception:
                            name_node_one = 'EEG 00' + str(row + 1)
                            pos_node_one = eeg_pos[name_node_one]
                        try:
                            pos_node_another = eeg_pos[name_node_another]
                        except Exception:
                            name_node_another = 'EEG 00' + str(col + 1)
                            pos_node_another = eeg_pos[name_node_another]
                        # print('x: {}, {}\ty: {}, {}'.format(pos_node_one[0], pos_node_another[0], pos_node_one[1], pos_node_another[1]))
                        # linewidth = abs(remedied_value)*3
                        linewidth_r = abs(remedied_value)*1
                        linewidth_o = abs(original_value)*1
                        # print('r {}, o {}, max_r {}, max_o {}'.
                        #       format(linewidth_r, linewidth_o, np.max(remedied), np.min(orginal)))
                        linewidth = linewidth_r - linewidth_o


                        # if(linewidth >=0.5):
                        #     color = '#59c362'
                        #     ax.plot([pos_node_one[0], pos_node_another[0]], [pos_node_one[1], pos_node_another[1]],
                        #             color=color, linewidth=linewidth*1.5)
                        #
                        # if(linewidth <= -0.5):
                        #     color = '#4cbafe'
                        #     # ax.plot([pos_node_one[0], pos_node_another[0]], [pos_node_one[1], pos_node_another[1]],
                        #     #         color=color, linewidth=linewidth*1.5)

                        # linewidth = abs(original_value)*1

                        # if(linewidth >= 0):
                        #     color='#7d7d7d'
                        #     # ax.plot([pos_node_one[0], pos_node_another[0]], [pos_node_one[1], pos_node_another[1]], color=color, linewidth=linewidth)
                        # if(linewidth <0):
                        #     color='black'
                        #     ax.plot([pos_node_one[0], pos_node_another[0]], [pos_node_one[1], pos_node_another[1]], color=color, linewidth=linewidth)

                        ave_dict[str(row) + ',' + str(col)].append(linewidth)
                        if (len(loc_dict.keys()) == 0):
                            loc_dict[str(row) + ',' + str(col)] = [[pos_node_one[0], pos_node_another[0]], [pos_node_one[1], pos_node_another[1]]]
                        elif (str(row) + ',' + str(col) in list(loc_dict.keys())):
                            pass
                        else:
                            loc_dict[str(row) + ',' + str(col)] = [[pos_node_one[0], pos_node_another[0]], [pos_node_one[1], pos_node_another[1]]]
        # plt.show()

        # plt.savefig(save_path +'/vis_'+str(i+1)+'_topology.jpg', dpi=450)

        # plt.savefig(save_path +'/diffo_'+str(i+1)+'_topology.jpg', dpi=450)

        # # plt.savefig('/Users/linya/Desktop/research_files/midterm/au_' + str(i+1)+'_topology.jpg', dpi=450)
        # plt.savefig('/Users/linya/Desktop/research_files/midterm/vis_' + str(i+1)+'_topology.jpg', dpi=450)
        # # plt.savefig('/Users/linya/Desktop/research_files/midterm/og_au_' + str(i+1)+'_topology.jpg', dpi=450)
        # # plt.savefig('/Users/linya/Desktop/research_files/midterm/og_vis_' + str(i+1)+'_topology.jpg', dpi=450)

        plt.gca()

print(ave_dict)
print('-'*20)
print(loc_dict)
ave_dict = {key: np.mean(np.array(value))for key, value in ave_dict.items()}
print(ave_dict)

read_mont = mne.channels.find_layout(info)

eeg_layout = mne.channels.make_eeg_layout(info)
returns = eeg_layout.plot()
plot_eeg_layout, eeg_pos, ax = returns[0], returns[1], returns[2]

for key, value in loc_dict.items():
    pre, post = value[0], value[1]
    linewidth = ave_dict[key]
    print(key, ', ', linewidth)
    if(linewidth >= 0.5):
        color = '#59c362'
        ax.plot(pre, post, color=color, linewidth=linewidth)
    if(linewidth <-0.5):
        color = '#4cbafe'
        # ax.plot(pre, post, color=color, linewidth=linewidth)
plt.savefig(save_path +'/ave_diff_aur_'+str(i+1)+'_topology.jpg', dpi=450)
