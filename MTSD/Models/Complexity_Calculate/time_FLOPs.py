from statistics import mode
import sys
sys.path.append('/home/lijun2/project/zyw_project/Py_Projects/MTSD/MTSD/Models')
import Model_Architecture
sys.path.append('/home/lijun2/project/zyw_project/Py_Projects/MTSD/MTSD/data_preprocess')
import data_entry

from ptflops import get_model_complexity_info
# from thop import profile

import torch
import torch.nn as nn

import os
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
subject_name = 'changcan'
window_num = 76
parent_path = '/home/lijun2/project/zyw_project/2nd_data'
file_name = 'single_data.npy'
raw_data_path = os.path.join(parent_path, file_name)
# 128, 500
raw_data = np.load(raw_data_path, allow_pickle=True)
raw_data = raw_data[..., np.newaxis]
print(raw_data.shape)

raw_data = np.ones((62, 800, 1))
colored_data = data_entry.Insert_color_channel(data1=raw_data, data2=raw_data, data3=raw_data, color=3)
colored_data = np.transpose(colored_data, (3, 0, 1, 2))
print('color dara {}'.format(colored_data.shape))
trials, color, height_t, width_t = colored_data.shape
copula_raw = np.ones((1, 1, 62, 62))

x = torch.from_numpy(colored_data)
x_corr = torch.from_numpy(copula_raw)
print('x shape {} and x_coor shape {}'.format(x.size(), x_corr.size()))
classes = 2

model = Model_Architecture.ECNN_entry(num_classes=classes,
                                      input_width=width_t, input_height=height_t,
                                      kernel_size=85, dilations=[1, 1, 1], planes=18,
                                      )

# model = Model_Architecture.ECNN_BASE_CM_SA(num_classes=classes,
#                                       input_width=width_t, input_height=height_t,
#                                       kernel_size=85, dilations=[1, 1, 1], planes=18,
#                                       attention_mode=True, attention_hint=0
#                                       )
# model = Model_Architecture.ECNN_Base_PCM_entry(num_classes=classes,
#                                       input_width=width_t, input_height=height_t,
#                                       kernel_size=85, dilations=[1, 2, 3], planes=18,
#                                       )

# model = Model_Architecture.MTSD_entry(num_classes=classes,
#                                       input_width=width_t, input_height=height_t,
#                                       kernel_size=85, dilations=[1, 2, 3], planes=18,
#                                       attention_mode=True, attention_hint=0)

model = model.to(device)
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)

#   hyperparameters
b_s = 16
lr = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
x = x.to(device)
x_corr = x_corr.to(device)
output = model(x.float())
with torch.no_grad():
    macs, params = get_model_complexity_info(model, (3, 62, 800), 
                                             as_strings=True, print_per_layer_stat=True, verbose=True)
    
    # macs, params = profile(model, inputs=(input, ), )
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    print('parameters:', sum(param.numel() for param in model.parameters()))


# import numpy as np
# import os

# main_name = '/Volumes/Seagate Backup Plus Drive/py_vac/AUS_File/data_76seg/76_EEG_data_changcan.npy'
# raw_data = np.load(main_name, allow_pickle=True)
# print(type(raw_data))

# print(raw_data.shape)
# single_image = raw_data[:, :, 0]
# save_path = '/Users/linya/Desktop/research_files/2nd'
# np.save(os.path.join(save_path, 'single_data.npy'), single_image)