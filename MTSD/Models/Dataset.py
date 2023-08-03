import torch
from torch.utils import data
import numpy as np

class My_Dataset(data.Dataset):

    def __init__(self, list_IDs, labels, train_data, totensor=False):
        self.list_IDs = list_IDs
        self.labels = labels
        self.train_data = train_data
        self.totensor = totensor

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        #   IDs -> list
        if(torch.is_tensor(index)):
            index = index.tolist()

        ID = self.list_IDs[index]
        x = self.train_data[ID, :, :, :]
        y = int(self.labels[ID])

        if(self.totensor):
            x = torch.from_numpy(x)
            y = torch.from_numpy(np.asarray(y))

        samples = {'batch_train': x, 'batch_label': y}

        return samples

#   insert color channel to raw data
def Insert_color_channel(data):
    print('-'*10, 'colored data', '-'*10)
    color = 1
    colored_data = []
    for c in range(color):
        colored_data.append(data)
    colored_data = np.asarray(colored_data)
    print(colored_data.shape, '\t', type(colored_data))

    return colored_data

class My_minxed_Dataset(data.Dataset):

    def __init__(self, list_IDs, labels, train_data, mask_data, totensor=False):
        self.list_IDs = list_IDs
        self.labels = labels
        self.train_data = train_data
        self.mask_data = mask_data
        self.totensor = totensor

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        #   IDs -> list
        if(torch.is_tensor(index)):
            index = index.tolist()

        ID = self.list_IDs[index]
        y = int(self.labels[ID])
        x = self.train_data[ID, :, :, :]
        x_copula = self.mask_data[ID, :, :, :]


        if(self.totensor):
            x = torch.from_numpy(x)
            y = torch.from_numpy(np.asarray(y))
            x_copula = torch.from_numpy(x_copula)

        samples = {'batch_train': x,
                   'batch_label': y,
                   'batch_mask': x_copula}

        return samples
