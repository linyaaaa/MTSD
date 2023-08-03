import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import mne


removed_electrodes = [127, 126, 17,
                          128, 48, 119, 125,
                          43, 49, 56, 63, 68, 73, 81, 88, 94, 99, 107, 113, 120]

print(len(removed_electrodes))

selected_electrodes = ['Fp2', 'Fz', 'Fp1', 'F3', 'F7', 'C3', 'T7', 'P3', 'P7', 'Pz', 'O1', 'Oz',
                       'O2', 'P4', 'P8', 'C4', 'F8', 'F4']
print(len(selected_electrodes))
excel_path = '/Volumes/Seagate Backup Plus Drive/py_vac/AUS_File/electrode_file/electrode_selection.xlsx'
excel = pd.read_excel(excel_path).values[:, 0]
print(excel.shape, '\n', excel)
kept = list(set(list(np.arange(len(excel))))-set(removed_electrodes))
kept_excel = excel[kept]
kept_excel = list(kept_excel)
selected_indexes = [kept_excel.index(s_e) for s_e in selected_electrodes]
print(len(selected_indexes))
print(selected_indexes)

names_subjects = np.load('/Volumes/Seagate Backup Plus Drive/py_vac/AUS_File/data_deleted_trials' + '/names_subjects.npy')
print(names_subjects)


def show(a, b, c, names):
    for i_index, i in enumerate(c):
        fig = plt.figure(num=i_index, figsize=(20, 20))
        ax = plt.gca()
        # , cmap="YlGn"
        im = ax.imshow(i)
        # create color bar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('dependence', rotation=-90, va="bottom")
        ax.set_xticks(np.arange(len(b)))
        ax.set_yticks(np.arange(len(a)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(b, fontsize=20)
        ax.set_yticklabels(a, fontsize=20)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        # We want to show all ticks...
        ax.set_xticks(np.arange(len(b) + 1) - .5, minor=True)
        ax.set_yticks(np.arange(len(a) + 1) - .5, minor=True)
        ax.spines[:].set_visible(False)
        # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        plt.savefig('/Users/linya/Desktop/research_files/2nd/revison/' + names[i_index] + '.jpg', dpi=450)
        plt.clf()
    # plt.show()


def normlize(matrix, trials):
    for t in np.arange(trials):
        temp = matrix[t, :, :]
        min_value, max_value = np.min(temp, axis=1), np.max(temp, axis=1)
        min_value, max_value = np.reshape(min_value, (min_value.shape[0], 1)), np.reshape(max_value, (max_value.shape[0], 1))
        temp = (temp - min_value) / (max_value - min_value)
        matrix[t, :, :] = temp
    return matrix


sub_vd, sub_nonvd = None, None
for name_index, name in enumerate(names_subjects):
    path = '/Volumes/Seagate Backup Plus Drive/py_vac/AUS_File/data_correlation_matrix/'
    prename = 'raw_correlation_matix_'
    name = name
    predix = '.npy'

    complete = os.path.join(path, prename+name+predix)
    corr_data = np.load(complete, allow_pickle=True)
    print(type(corr_data), ', ', corr_data.shape)

    window_num = 76
    path_label = '/Volumes/Seagate Backup Plus Drive/py_vac/AUS_File/'
    sub_label = np.load(path_label + 'data_' + str(window_num) + 'seg/' + str(window_num) + 'data_label_new.npy', allow_pickle=True).item()
    label = sub_label[name]
    print('label {}, {}'.format(type(label), label.shape))

    trials = label.shape[0]
    vd_base_label = label[label == 1]
    nonvd_base_label = label[label == 0]

    matrix_vd = corr_data[0, :, :, vd_base_label]
    matrix_nonvd = corr_data[0, :, :, nonvd_base_label]
    print(matrix_vd.shape, ', ', matrix_nonvd.shape)
    vd_trials, nonvd_trials = matrix_vd.shape[0], matrix_nonvd.shape[0]
    matrix_vd = normlize(matrix_vd, vd_trials)
    matrix_nonvd = normlize(matrix_nonvd, nonvd_trials)

    #'''
    matrix_vd[np.abs(matrix_vd)<=0.3], matrix_nonvd[np.abs(matrix_nonvd)<=0.3] = 0, 0
    matrix_vd[np.abs(matrix_vd)>0.3], matrix_nonvd[np.abs(matrix_nonvd)>0.3] = 1, 1

    vd_sum, nonvd_sum = np.sum(matrix_vd, axis=0), np.sum(matrix_nonvd, axis=0)
    mean_vd, mean_nonvd = vd_sum/vd_trials, nonvd_sum/nonvd_trials
    print(vd_trials, ', ', nonvd_trials)
    if(name_index == 0):
        sub_vd, sub_nonvd = mean_vd, mean_nonvd
    else:
        sub_vd = sub_vd + mean_vd
        sub_nonvd = sub_nonvd + mean_nonvd
    # break
num = len(names_subjects)
# num = 1
mean_vd, mean_nonvd = sub_vd/num, sub_nonvd/num

orig_vd, orig_nonvd = mean_vd, mean_nonvd
mean_vd[np.abs(mean_vd<=0.3)] = 0
mean_vd[np.abs(mean_vd>0.3)] = 1
mean_nonvd[np.abs(mean_nonvd<=0.3)] = 0
mean_nonvd[np.abs(mean_nonvd>0.3)] = 1
mean_vd = mean_vd[selected_indexes, :]
mean_vd = mean_vd[:, selected_indexes]
mean_nonvd = mean_nonvd[selected_indexes, :]
mean_nonvd = mean_nonvd[:, selected_indexes]


# names = ['vd', 'nonvd']
names = ['diff']
diff_ = orig_vd - orig_nonvd

# show(selected_electrodes,selected_electrodes, [diff_], names)
# show(list(kept_excel),list(kept_excel), [diff_], names)


biosemi_montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
print(biosemi_montage)
fig, returns = biosemi_montage.plot(kind='topomap', show_names=True)
rows, columns = diff_.shape
pos, ch_names = returns[0], returns[1]
print(pos, '\n', ch_names)
plt.gca()
# plt.plot(pos[0, :], pos[1,:])
for row in range(rows):
    for col in np.arange(columns):
        if(diff_[row, col] > 0):
            start, end = row, col
            start_pos_left, start_pos_right = pos[start, 0], pos[start, 1]
            end_pos_left, end_pos_right = pos[end, 0], pos[end, 1]
            plt.scatter([start_pos_left, end_pos_left], [start_pos_right, end_pos_right], c='#d6604d', s=27)
            plt.plot([start_pos_left, end_pos_left], [start_pos_right, end_pos_right], color='#d6604d', linewidth=2)
            # pass
        elif(diff_[row, col] < 0):
            # start, end = row, col
            # start_pos_left, start_pos_right = pos[start, 0], pos[start, 1]
            # end_pos_left, end_pos_right = pos[end, 0], pos[end, 1]
            # plt.plot([start_pos_left, end_pos_left], [start_pos_right, end_pos_right], color='#2b8cbe', linewidth=2)
            # plt.scatter([start_pos_left, end_pos_left], [start_pos_right, end_pos_right], c='#2b8cbe', s=27)

            pass
        else:
            pass

save_names = ['vd.jpg', 'nonvd.jpg']
plt.savefig('/Users/linya/Desktop/research_files/thesis/'+save_names[0], dpi=450)
# plt.show()
