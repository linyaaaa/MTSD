'''data preprocess
ATTENTION: size of the preprocessed data can exceed 690GB'''

import numpy as np
import scipy.io as io
import os
# from PyEMD import EMD
# from copulas.multivariate.gaussian import GaussianMultivariate

def Insert_color_channel(data1, data2, data3, color):
    imfs = [data1, data2, data3]
    color = color
    colored_data = []
    for c in range(color):
        colored_data.append(imfs[c])
    colored_data = np.array(colored_data)
    return colored_data

# def Normalization_Data(data):
#     channels, datapoints, trials = data.shape
#     new_data = np.zeros((channels, datapoints, trials))
#     for trial in range(trials):
#         for sin_channel in range(channels):
#             temp_mean = np.mean(data[sin_channel, :, trial], axis=0)
#             temp_std = np.std(data[sin_channel, :, trial], axis=0, ddof=1)
#             new_data[sin_channel, :, trial] = (data[sin_channel, :, trial] - temp_mean)/temp_std
#     return new_data

# '''feature extraction'''
# def EMD_Signal(path, window_number):
#     window_num = window_number
#     for i, sub in enumerate(names_subjects):
#         emd = EMD()
#         data = np.load(path + 'data_' + str(window_num) + 'seg/' + str(window_num) + '_EEG_data_' + sub + '.npy')
#         channels, datapoints, trials = data.shape
#         new_data = Normalization_Data(data)
#         imf1 = np.zeros((channels, datapoints, trials))
#         imf2 = np.zeros((channels, datapoints, trials))
#         imf3 = np.zeros((channels, datapoints, trials))
#         for trial in range(trials):
#             for sin_channel in range(channels):
#                 temp = new_data[sin_channel, :, trial]
#                 IMFs = emd.emd(S=temp, T=np.arange(500))
#                 try:
#                     imf1[sin_channel, :, trial] = IMFs[0, :]
#                     imf2[sin_channel, :, trial] = IMFs[1, :]
#                     imf3[sin_channel, :, trial] = IMFs[2, :]
#                 except Exception:
#                     print(sin_channel, '\t', trial)

#         np.save(path + 'data_' + str(window_num) + 'seg/imf1_data_' + sub + '.npy', imf1)
#         np.save(path + 'data_' + str(window_num) + 'seg/imf2_data_' + sub + '.npy', imf2)
#         np.save(path + 'data_' + str(window_num) + 'seg/imf3_data_' + sub + '.npy', imf3)

#     return True


# def Raw_Data_Segment(path, sub_label):
#     '''following should be modified'''
#     # pre_name = 'data/EEG_data_'

#     '''set your own path for saving generated EEG data'''
#     path_save = ''

#     sampling_frequency = 500
#     baseline_period = 0.5
#     window_length = 1
#     step = float(10) / 500
#     for sub in names_subjects:
#         sub_data = io.loadmat(path + '/' + sub + '.mat')['a']
#         #   keep the convinced data
#         start_point = int(baseline_period * sampling_frequency)
#         sub_data = np.delete(sub_data, np.arange(0, start_point), axis=1)
#         channels, data_points, trials = sub_data.shape
#         number_window = int((2.5 - window_length) / step + 1)
#         window_data = int(window_length * sampling_frequency)
#         step_data = int(step * sampling_frequency)

#         new_data = np.zeros(shape=(channels, window_data, int(trials * number_window)))
#         #   sliding window
#         for trial in range(trials):
#             temp = sub_data[:, :, trial]
#             for window in range(number_window):
#                 start_point = 0 * window_data + window * step_data
#                 end_point = (0 + 1) * window_data + window * step_data
#                 if (end_point - start_point != window_data):
#                     raise Exception('wrong in start and end point')
#                 else:
#                     pass
#                 temp_window_data = temp[:, start_point: end_point]
#                 new_data[:, :, trial * number_window + window] = temp_window_data

#         if (os.path.exists(path + '/data_' + str(number_window) + 'seg/')):
#             pass
#         else:
#             os.makedirs(path + '/data_' + str(number_window) + 'seg/')
#         #   save new data matrix
#         np.save(path + '/data_' + str(number_window) + 'seg/' + str(number_window) + '_EEG_data_' + sub + '.npy',
#                 new_data)
#         temp_label = np.array([[sin_label] * number_window for sin_label in sub_label[sub]])
#         sub_label[sub] = np.ravel(temp_label)

#     np.save(path + '/data_' + str(number_window) + 'seg/' + str(number_window) + 'data_label_new.npy', sub_label)

#     return number_window

# '''copula part'''
# def MyCopula(each_color, t, data):
#     gc = GaussianMultivariate()
#     exceptions_list = []
#     copula_data = gc.fit(data, exceptions_list)

#     if(len(copula_data) == 0):
#         correlation_matrix = gc.covariance
#         original_matrix = gc.original_correlation
#         return correlation_matrix, original_matrix
#     else:
#         return [each_color, t]

# def copula_entry(path, window_number):
#     '''remove marginal electrodes because signals on them are unstable'''
#     removed_electrodes = [127, 126, 17,
#                           128, 48, 119, 125,
#                           43, 49, 56, 63, 68, 73, 81, 88, 94, 99, 107, 113, 120]
#     removed_electrodes = list(np.array(removed_electrodes) - 1)
#     all_electrodes = list(np.arange(0, 128))
#     picked_electodes = list(set(all_electrodes) - set(removed_electrodes))
#     window_num = window_number

#     for i, sub in enumerate(names_subjects):
#         raw_data = np.load(path + 'data_' + str(window_num) + 'seg/' + str(window_num) + '_EEG_data_' + sub + '.npy')
#         raw_data = raw_data[picked_electodes, :]
#         colored_data = Insert_color_channel(data1=raw_data, data2=0, data3=0, color=1)
#         colors, channels, datapoints, trials = colored_data.shape
#         colored_correlation = []
#         exceptions = {}
#         for each_color_, each_color in enumerate(colored_data):
#             copula_matrix = np.zeros((channels, channels, trials))
#             exceptions[each_color_] = []

#             for t in range(trials):
#                 data = each_color[:, :, t].T
#                 returns = MyCopula(each_color_, t, data)
#                 if (type(returns) == list):
#                     exceptions[each_color_].append(t)
#                 else:
#                     correlation_matrix, original_matrix = returns[0], returns[1]
#                     copula_matrix[:, :, t] = correlation_matrix
#             colored_correlation.append(copula_matrix)

#         new_exceptions = []
#         for color_, color_content in enumerate(exceptions):
#             if (len(exceptions[color_content]) == 0):
#                 pass
#             else:
#                 new_exceptions = list(set(new_exceptions).union(set(exceptions[color_content])))
#         if (len(new_exceptions) == 0):
#             pass
#         else:
#             kept_dims = list(set(list(np.arange(trials))) - set(new_exceptions))
#             for color_, color_content in enumerate(colored_correlation):
#                 colored_correlation[color_] = color_content[:, :, kept_dims]
#         colored_correlation = np.array(colored_correlation)

#         '''create your own folder to save the generated correlation matrix'''
#         if(os.path.exists(path + '/correlation_matrix/')):
#             pass
#         else:
#             os.makedirs(path + '/correlation_matrix/')
#         np.save(path + '/correlation_matrix/raw_correlation_matix_' + sub + '.npy', colored_correlation)
#         if(len(new_exceptions) == 0):
#             pass
#         else:
#             np.save(path + 'correlation_matrix/raw_correlation_matix_' + sub + '_excep.npy', np.array(new_exceptions))


# def EMD_COMBINATION(remainig, path, sub_label, window_num, color):

#     for i, sub in enumerate(remainig):
#         imf1 = np.load(path + 'data_' + str(window_num) + 'seg/imf1_data_' + sub + '.npy')
#         imf2 = np.load(path + 'data_' + str(window_num) + 'seg/imf2_data_' + sub + '.npy')
#         imf3 = np.load(path + 'data_' + str(window_num) + 'seg/imf3_data_' + sub + '.npy')

#         labels = sub_label[sub]
#         colored_data = Insert_color_channel(imf1, imf2, imf3, color)
#         data = np.transpose(colored_data, (3, 0, 1, 2))
#         if( i == 0 ):
#             combined_data = data
#             combined_label = labels
#         else:
#             combined_data = np.vstack((combined_data, data))
#             combined_label = np.hstack((combined_label, labels))
#     return combined_data, combined_label

# if __name__ == '__main__':

#     '''set your own path of downloaded data from google drive'''
#     path = ''
#     names_subjects = np.load(path + '/names_subjects.npy')
#     sub_label = np.load(path + '/data_label.npy', allow_pickle=True).item()

#     window_number = Raw_Data_Segment(path, sub_label)
#     EMD_Signal(path, window_number)
#     copula_entry(path, window_number)



