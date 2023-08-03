'''
norm_copula: normalized + threshold_copula + model
'''
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import random
from torch.utils import data as torch_data
import torch
import torch.nn as nn
from data_preprocess import data_entry
from Models import Dataset, Model_Architecture

colors = ['#d50000', '#c51162', '#aa00ff', '#6200ea', '#304ffe', '#0091ea', '#00b8d4', '#00bfa5',
          '#00c853', '#64dd17', '#aeea00', '#ffd600', '#ffab00', '#ff6d00', '#dd2c00', '#3e2723',
          '#212121', '#263238', '#ff5252', '#ff4081', '#e040fb', '#7c4dff', '#536dfe', '#40c4ff',
          '#18ffff', '#64ffda', '#69f0ae', '#b2ff59', '#eeff41', '#ffff00', '#ffd740', '#ffab40',
          '#ff6e40', '#5d4037', '#616161', '#455a64', '#ff8a80', '#ff80ab', '#ea80fc', '#b388ff',
          '#8c9eff', '#80d8ff', '#84ffff', '#a7ffeb', '#b9f6ca', '#ccff90', '#f4ff81', '#ffff8d']
device = 'cuda' if torch.cuda.is_available() else 'cpu'


model = Model_Architecture.MTSD_entry(num_classes=2,
                                      input_width=500,
                                      input_height=108,
                                      kernel_size=85,
                                      dilations=[1, 2, 3],
                                      planes=18,
                                      attention_mode=True,
                                      attention_hint=0)



print('parameters:', sum(param.numel() for param in model.parameters()))

print(model)
for param in model.parameters():
    print(param.numel())