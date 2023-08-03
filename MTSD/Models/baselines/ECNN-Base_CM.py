import numpy as np
from torch.utils import data as torch_data
import torch
import torch.nn as nn
import random
import os
from data_preprocess import data_entry
from Models import Dataset, Model_Architecture


window_num = 76
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''channel selection'''
path = ''
names_subjects = list(np.load(path + '/names_subjects.npy'))
sub_label = np.load(path + '/data_' + str(window_num) + 'seg/' + str(window_num) + 'data_label_new.npy', allow_pickle=True).item()
removed_electrodes = [127, 126, 17,
                      128, 48, 119, 125,
                      43, 49, 56, 63, 68, 73, 81, 88, 94, 99, 107, 113, 120]
removed_electrodes = list(np.array(removed_electrodes) - 1)
all_electrodes = list(np.arange(0, 128))
picked_electodes = list(set(all_electrodes) - set(removed_electrodes))

for sub_index, sub in enumerate(names_subjects):
    imf1 = np.load(path + '/data_' + str(window_num) + 'seg/imf1_data_' + sub + '.npy')
    imf2 = np.load(path + '/data_' + str(window_num) + 'seg/imf2_data_' + sub + '.npy')
    imf3 = np.load(path + '/data_' + str(window_num) + 'seg/imf3_data_' + sub + '.npy')
    label = sub_label[sub]
    imf1 = imf1[picked_electodes, :, :]
    imf2 = imf2[picked_electodes, :, :]
    imf3 = imf3[picked_electodes, :, :]

    colored_data = data_entry.Insert_color_channel(data1=imf1, data2=imf2, data3=imf3, color=3)
    colored_data = np.transpose(colored_data, (3, 0, 1, 2))
    trials, color, height, width = colored_data.shape

    train_trials = int(0.8 * trials)
    train_index = random.sample(list(np.arange(trials)), train_trials)
    test_index = list(set(list(np.arange(trials))) - set(train_index))
    train_data = colored_data[train_index, :, :, :]
    test_data = colored_data[test_index, :, :, :]
    train_label = label[train_index]
    test_label = label[test_index]

    trials_t, color_t, height_t, width_t = train_data.shape
    training_data = Dataset.My_Dataset(list_IDs=list(np.arange(trials_t)),
                                       labels=train_label,
                                       train_data=train_data,
                                       totensor=True)
    #   hyperparameters
    b_s = 16
    b_s_t = 128
    classes = 2
    e_p = 50
    lr = 0.001
    SHOW_LOSS = 16
    training_loader = torch_data.DataLoader(dataset=training_data,
                                            batch_size=b_s,
                                            shuffle=True)

    t_trials, t_colors, t_height, t_width = test_data.shape
    testing_data = Dataset.My_Dataset(list_IDs=list(np.arange(t_trials)),
                                      labels=test_label,
                                      train_data=test_data,
                                      totensor=True)
    testing_loader = torch_data.DataLoader(dataset=testing_data,
                                           batch_size=b_s_t,
                                           shuffle=False)

    model = Model_Architecture.ECNN_Base_CM_entry(num_classes=classes,
                                                  input_width=width_t,
                                                  input_height=height_t,
                                                  kernel_size=85,
                                                  dilations=[1, 1, 1],
                                                  planes=18)

    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    ave_train_acc = []
    ave_train_loss = []
    ave_test_acc = []
    for epoch in range(e_p):
        num_correct_train_record = 0
        loss_train_record = 0
        train_record = 0
        for step, sample_batch in enumerate(training_loader):
            b_x = sample_batch['batch_train']
            b_y = sample_batch['batch_label']
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            output = model(b_x.float())
            loss = loss_function(output.float(), b_y.long())

            #   optimizer process
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train_record = loss_train_record + loss.item()
            _record, pred_record = torch.max(output, 1)
            num_correct_train_record = num_correct_train_record + (pred_record == b_y).sum().item()
            train_record = train_record + b_x.size(0)

            if ((step + 1) % SHOW_LOSS == 0):
                _, pred = torch.max(output, 1)
                loss_train = loss
                num_correct_train = (pred == b_y).sum().item()
                print('subs\t', 'train \tEPOCH|{}'.format(epoch + 1), '\tstep|{}'.format(step + 1),
                      '\ttrain acc|{:.5f}'.format(float(num_correct_train) / b_y.size(0)),
                      '\ttrain loss|{:.5f}'.format(loss_train.item()))
        ave_train_acc.append(float(num_correct_train_record) / train_record)
        ave_train_loss.append(loss_train_record)

        with torch.no_grad():
            num_correct_test = 0
            num_test = 0
            for step_t, sample_batch_t in enumerate(testing_loader):
                b_x_t = sample_batch_t['batch_train']
                b_y_t = sample_batch_t['batch_label']
                b_x_t = b_x_t.to(device)
                b_y_t = b_y_t.to(device)
                test_output = model(b_x_t.float())
                _t, pred_t = torch.max(test_output, 1)
                num_correct_test = (pred_t == b_y_t).sum().item() + num_correct_test
                num_test = num_test + b_y_t.size(0)
            print(sub, ' test\t', 'test acc|{:.5f}'.format(float(num_correct_test) / num_test))

        ave_test_acc.append(float(num_correct_test) / num_test)

        if(epoch + 1 == 50):
            #   save model
            if (os.path.exists(path + '/model_parameters/')):
                pass
            else:
                os.makedirs(path + '/model_parameters/')
            torch.save(model.state_dict(), path + '/model_parameters/'
                       + 'ECNN_Base_CM_' + str(b_s) + '_' + str(lr) + str(window_num) + '_'
                       + sub + '.pth')
        else:
            pass
    if(os.path.exists(path + '/model_parameters/')):
        pass
    else:
        os.makedirs(path + '/model_parameters/')

    np.save(path + '/model_parameters/ECNN_Base_CM_' + str(b_s) + '_' + str(lr) + str(window_num) + '_' + sub + '_train_acc.npy', np.array(ave_train_acc))
    np.save(path + '/model_parameters/ECNN_Base_CM_' + str(b_s) + '_' + str(lr) + str(window_num) + '_' + sub + '_train_loss.npy', np.array(ave_train_loss))
    np.save(path + '/model_parameters/ECNN_Base_CM_' + str(b_s) + '_' + str(lr) + str(window_num) + '_' + sub + '_test_acc.npy', np.array(ave_test_acc))

    print('parameters:', sum(param.numel() for param in model.parameters()))
