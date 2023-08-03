import numpy as np
import os
from torch.utils import data as torch_data
import torch
import torch.nn as nn
import random 


import my_dataset, MTS_DGCHN, my_utils


device = 'cuda' if torch.cuda.is_available() else 'cpu'


data_parent_path = ''
processed_name = ''
# l_matrix == a_hat
l_matrix_name = ''

sub_names = [i for i in os.listdir(os.path.join(data_parent_path, processed_name))]


time = 1
for sub_name in sub_names:
    print('sub name ', sub_name)
    try:
        l_matrix = np.load(os.path.join(data_parent_path, l_matrix_name, sub_name, 'l_matrix.npy'), allow_pickle=True)
    except Exception:
        print('l matrix does not exist in {}'.format(sub_name))
        continue
    
    processed_data = np.load(os.path.join(data_parent_path, processed_name, sub_name, 'processed_data.npy'), allow_pickle=True)
    processed_label = np.load(os.path.join(data_parent_path, processed_name, sub_name, 'processed_label.npy'), allow_pickle=True)

    channel, datapoints, trials = processed_data.shape

    ratio = 0.8
    train_num = int(trials*0.8)
    test_num = int(trials - train_num)
    
    train_index = random.sample(list(np.arange(trials)), train_num)
    test_index = list(set(list(np.arange(trials))) - set(train_index))
    train_data = processed_data[:,:,train_index]
    test_data = processed_data[:,:,test_index]

    train_data = np.reshape(train_data, (train_num, channel, datapoints))
    test_data = np.reshape(test_data, (test_num, channel, datapoints))

    expand_train_data = train_data[:,np.newaxis,:,:]
    expand_test_data = test_data[:,np.newaxis,:,:]
    print('expand before {} and after {}'.format(train_data.shape, expand_train_data.shape))

    print('all label shape ', processed_label.shape)
    train_label = processed_label[train_index]
    test_label = processed_label[test_index]
    
    train_dataset = my_dataset.My_Dataset(list_IDs=list(np.arange(0, train_num)),
                                          labels=train_label,
                                          train_data=expand_train_data,
                                          totensor=True)
    test_dataset = my_dataset.My_Dataset(list_IDs=list(np.arange(0, test_num)),
                                          labels=test_label,
                                          train_data=expand_test_data,
                                          totensor=True)
    #   hyperparameters
    b_s = 32
    b_s_t = 128
    classes = 2
    e_p = 50
    lr = 0.001
    SHOW_LOSS = 10

    lambda_theta = 0.00001
    u_a = 0.2

    training_loader = torch_data.DataLoader(dataset=train_dataset, batch_size=b_s, shuffle=True)
    testing_loader = torch_data.DataLoader(dataset=test_dataset, batch_size=b_s_t, shuffle=False)

    model = MTS_DGCHN.MTS_DGCHN(2, (1, 108, 256), 12, 18, 192, 128,  43, 512, 0.2, 
                                l_matrix)

    model = model.to(device)

    loss_cross = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    early_stopping = my_utils.EarlyStopping(patience=7, verbose=True, delta=0.001, path='', parent_path=os.path.join(data_parent_path, 'checkpoint', sub_name))


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
            loss_term1 = loss_cross(output.float(), b_y.long())

            loss_term2 = sum(abs(p).sum() for p in model.parameters())

            loss_term3 = torch.linalg.norm(model.Get_S.GCN1.A_hat, 1)+torch.linalg.norm(model.Get_S.GCN2.A_hat, 1)+torch.linalg.norm(model.Get_S.GCN3.A_hat, 1)

            loss = loss_term1 + lambda_theta*loss_term2 + u_a*loss_term3

            #   optimizer process
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train_record = loss_train_record + loss.item()
            _record, pred_record = torch.max(output, 1)
            num_correct_train_record = num_correct_train_record + (pred_record == b_y).sum().item()
            train_record = train_record + b_x.size(0)

            if ((step + 1) % SHOW_LOSS == 0):
                print(sub_name,'\t', 'train \tEPOCH|{}'.format(epoch + 1), '\tstep|{}'.format(step + 1),
                        '\ttrain acc|{:.5f}'.format(float(num_correct_train_record) / train_record),
                        '\ttrain loss|{:.5f}'.format(loss_train_record / (step+1)))

        ave_train_acc.append(float(num_correct_train_record) / train_record)
        ave_train_loss.append(loss_train_record)

        early_stopping(loss_train_record/len(training_loader), model, str(epoch)+'.pth',optimizer)
        print('-' * 25, ' {} Early stopping '.format(early_stopping.early_stop), '-' * 25)

        if early_stopping.early_stop:
            print('-' * 25, ' {} is stopping '.format(early_stopping.early_stop), '-' * 25)

            checkpoint = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}
            checkpoint_path = os.path.join(data_parent_path, 'checkpoint', sub_name)
            stop_path = os.path.join(checkpoint_path, str(lr)+'lr'+str(time)+'time_'+'stop_' + str(epoch + 1) + 'epoch.pth')
            try:
                torch.save(checkpoint, stop_path)
            except Exception:
                os.makedirs(checkpoint_path)
            torch.save(checkpoint, stop_path)

            np.save(os.path.join(checkpoint_path, str(lr)+'lr'+str(time)+'time_'+str(epoch + 1) + 'epoch_train_class.npy'), ave_train_acc)
            np.save(os.path.join(checkpoint_path, str(lr)+'lr'+str(time)+'time_'+str(epoch + 1) + 'epoch_train_loss.npy'), ave_train_loss)

            '''test part'''
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

                print(sub_name, ' test\t', 'test acc|{:.5f}'.format(float(num_correct_test) / num_test))
                ave_test_acc.append(float(num_correct_test) / num_test)

                np.save(os.path.join(checkpoint_path, str(lr)+'lr'+str(time)+'time_'+str(epoch + 1) + 'epoch_test_class.npy'), ave_test_acc)

                print('parameters:', sum(param.numel() for param in model.parameters()))
            
            break

        elif((not early_stopping.early_stop) and ((epoch + 1) % 10 == 0)):
            '''test part'''
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

                print(sub_name, ' test\t', 'test acc|{:.5f}'.format(float(num_correct_test) / num_test))
                ave_test_acc.append(float(num_correct_test) / num_test)

                test_path = os.path.join(data_parent_path, 'test_result', sub_name)
                if(os.path.exists(test_path)):
                    pass
                else:
                    os.makedirs(test_path)

                np.save(os.path.join(test_path, str(lr)+'lr'+str(time)+'time_'+str(epoch + 1) + 'epoch_test_class.npy'), ave_test_acc)

            train_path = os.path.join(data_parent_path, 'train_result', sub_name)
            if(os.path.exists(train_path)):
                pass
            else:
                os.makedirs(train_path)
            np.save(os.path.join(train_path, str(lr)+'lr'+str(time)+'time_'+str(epoch + 1) + 'epoch_train_class.npy'), ave_train_acc)
            np.save(os.path.join(train_path, str(lr)+'lr'+str(time)+'time_'+str(epoch + 1) + 'epoch_train_loss.npy'), ave_train_loss)
        
        '''test part'''
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

            print(sub_name, ' test\t', 'test acc|{:.5f}'.format(float(num_correct_test) / num_test))
            ave_test_acc.append(float(num_correct_test) / num_test)

            test_path = os.path.join(data_parent_path, 'test_result', sub_name)
            if(os.path.exists(test_path)):
                pass
            else:
                os.makedirs(test_path)

            np.save(os.path.join(test_path, str(lr)+'lr'+str(time)+'time_'+str(epoch + 1) + 'epoch_test_class.npy'), ave_test_acc)

    print('# generator parameters:', sum(param.numel() for param in model.parameters()))
    
























