import torch.nn as nn
import torch
import math as m
from torch.nn import init
import numpy as np


'''baseline part'''
class ESTCNN(nn.Module):
    def __init__(self, num_classes, input_width, input_height, kernel_size, dilations, planes):
        super(ESTCNN, self).__init__()
        self.relu = nn.ReLU()
        self.blocks1 = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=(1, kernel_size), stride=1,
                      padding=0),
            self.relu,
            nn.BatchNorm2d(planes),
            nn.Conv2d(planes, planes, kernel_size=(1, kernel_size), stride=1,
                      padding=0),
            self.relu,
            nn.BatchNorm2d(planes),
            nn.Conv2d(planes, planes, kernel_size=(1, kernel_size), stride=1,
                      padding=0),
            self.relu,
            nn.BatchNorm2d(planes),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        )
        input_width = int((input_width - 2*3) / 2)

        self.blocks2 = nn.Sequential(
            nn.Conv2d(planes, 2*planes, kernel_size=(1, kernel_size), stride=1,
                      padding=0),
            self.relu,
            nn.BatchNorm2d(2*planes),
            nn.Conv2d(2*planes, 2*planes, kernel_size=(1, kernel_size), stride=1,
                      padding=0),
            self.relu,
            nn.BatchNorm2d(2*planes),
            nn.Conv2d(2*planes, 2*planes, kernel_size=(1, kernel_size), stride=1,
                      padding=0),
            self.relu,
            nn.BatchNorm2d(2*planes),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        )
        input_width = int((input_width - 2*3) / 2)

        self.blocks3 = nn.Sequential(
            nn.Conv2d(2*planes, 4 * planes, kernel_size=(1, kernel_size), stride=1,
                      padding=0),
            self.relu,
            nn.BatchNorm2d(4*planes),
            nn.Conv2d(4 * planes, 4 * planes, kernel_size=(1, kernel_size), stride=1,
                      padding=0),
            self.relu,
            nn.BatchNorm2d(4*planes),
            nn.Conv2d(4 * planes, 4 * planes, kernel_size=(1, kernel_size), stride=1,
                      padding=0),
            self.relu,
            nn.BatchNorm2d(4*planes),
            nn.MaxPool2d(kernel_size=(1, 7), stride=(1, 7))
        )
        input_width = int(((input_width - 2*3) - 7) / 7) + 1

        self.fc1 = nn.Linear(4*planes*input_width*input_height, 50)
        self.classifier = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.classifier(x)

        return x

def ESTCNN_entry(num_classes, input_width, input_height, kernel_size, dilations, planes):
    model = ESTCNN(num_classes, input_width, input_height, kernel_size, dilations, planes)
    return model


class ECNN(nn.Module):
    def __init__(self, num_classes, input_width, input_height, kernel_size, dilations, planes):
        super(ECNN, self).__init__()
        self.planes = planes
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv2d(3, planes, kernel_size=(1, self.kernel_size), stride=(1, 2),
                               padding=(0, 0))
        self.temp = int((input_width - self.kernel_size)/2) + 1

        self.conv2 = nn.Conv2d(planes, 2*planes, kernel_size=(1, self.kernel_size),
                                  stride=(1, 2),
                                  padding=(0, 0))

        self.temp = int((self.temp - self.kernel_size)/2) + 1
        self.conv3 = nn.Conv2d(2*planes, 3*2*planes, kernel_size=(1, self.temp), stride=(1, 1),
                               padding=(0, 0))
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(3*1*input_height*2*planes, 4*planes)
        self.fc2 = nn.Linear(4*planes, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        temp = self.conv2(x)
        x = self.relu(temp)
        x = self.conv3(x)
        x = self.relu(x)
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

def ECNN_entry(num_classes, input_width, input_height, kernel_size, dilations, planes):
    model = ECNN(num_classes, input_width, input_height, kernel_size, dilations, planes)
    return model


class ECNN_Base_CM(nn.Module):
    def __init__(self, num_classes, input_width, input_height, kernel_size, dilations, planes):
        super(ECNN_Base_CM, self).__init__()
        self.planes = planes

        self.conv1 = nn.Conv2d(3, planes, kernel_size=(1, kernel_size), stride=(1, 2),
                               padding=(0,  0))
        self.temp = int((input_width - kernel_size)/2) + 1

        self.paddings = [(0, 0), (0, 15), (0, 25)]
        self.kernel_size_new = [85, 115, 135]
        self.conv2_l1 = nn.Conv2d(planes, 2*planes, kernel_size=(1, self.kernel_size_new[0]),
                                  stride=(1, 2), padding=self.paddings[0], dilation=(1, dilations[0]))
        self.conv2_l2 = nn.Conv2d(planes, 2*planes, kernel_size=(1, self.kernel_size_new[1]),
                                  stride=(1, 2), padding=self.paddings[1], dilation=(1, dilations[1]))
        self.conv2_l3 = nn.Conv2d(planes, 2*planes, kernel_size=(1, self.kernel_size_new[2]),
                                  stride=(1, 2), padding=self.paddings[2], dilation=(1, dilations[2]))
        self.maxpool2 = nn.MaxPool2d((1, 2), stride=(1, 2))
        self.temp = int((self.temp - self.kernel_size_new[0])/2) + 1

        self.conv3 = nn.Conv2d(3*2*planes, 3*2*planes, kernel_size=(1, self.temp), stride=(1, 1),
                               padding=(0, 0))

        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(3*1*input_height*2*planes, 4*planes)
        self.fc2 = nn.Linear(4*planes, num_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        temp = self.conv2_l1(x)
        multiscale_x_1 = self.relu(temp)
        temp = self.conv2_l2(x)
        multiscale_x_2 = self.relu(temp)
        temp = self.conv2_l3(x)
        multiscale_x_3 = self.relu(temp)
        x = torch.cat((multiscale_x_1, multiscale_x_2, multiscale_x_3), dim=1)
        x = self.conv3(x)
        x = self.relu(x)
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

def ECNN_Base_CM_entry(num_classes, input_width, input_height, kernel_size, dilations, planes):
    model = ECNN_Base_CM(num_classes, input_width, input_height, kernel_size, dilations, planes)
    return model


class ECNN_Base_PCM(nn.Module):
    def __init__(self, num_classes, input_width, input_height, kernel_size, dilations, planes):
        super(ECNN_Base_PCM, self).__init__()
        self.planes = planes

        self.conv1 = nn.Conv2d(3, planes, kernel_size=(1, kernel_size), stride=(1, 2),
                               padding=(0,  0))
        self.temp = int((input_width - kernel_size)/2) + 1

        self.paddings = [(0, 0), (0, 0), (0, 0)]
        self.kernel_size_new = [85, 43, 29]

        self.conv2_l1 = nn.Conv2d(planes, 2*planes, kernel_size=(1, self.kernel_size_new[0]),
                                  stride=(1, 2), padding=self.paddings[0], dilation=(1, dilations[0]))
        self.conv2_l2 = nn.Conv2d(planes, 2*planes, kernel_size=(1, self.kernel_size_new[1]),
                                  stride=(1, 2), padding=self.paddings[1], dilation=(1, dilations[1]))
        self.conv2_l3 = nn.Conv2d(planes, 2*planes, kernel_size=(1, self.kernel_size_new[2]),
                                  stride=(1, 2), padding=self.paddings[2], dilation=(1, dilations[2]))
        self.maxpool2 = nn.MaxPool2d((1, 2), stride=(1, 2))
        self.temp = int((self.temp - self.kernel_size_new[0])/2) + 1

        self.conv3 = nn.Conv2d(2*planes, 2*planes, kernel_size=(1, self.temp), stride=(1, 1), padding=(0, 0))

        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(1*input_height*2*planes, 4*planes)
        self.fc2 = nn.Linear(4*planes, num_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        temp = self.conv2_l1(x)
        multidilation_x_1 = self.relu(temp)

        temp = self.conv2_l2(x)
        multidilation_x_2 = self.relu(temp)
        multidilation_x_1 = multidilation_x_1 + multidilation_x_2

        temp = self.conv2_l3(x)
        multidilation_x_3 = self.relu(temp)
        multidilation_x_1 = multidilation_x_1 + multidilation_x_3
        x = multidilation_x_1

        x = self.conv3(x)
        x = self.relu(x)

        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

def ECNN_Base_PCM_entry(num_classes, input_width, input_height, kernel_size, dilations, planes):
    model = ECNN_Base_PCM(num_classes, input_width, input_height, kernel_size, dilations, planes)
    return model

'''ablation part'''
class SelfAttention(nn.Module):
    def __init__(self, input_planes):
        super(SelfAttention, self).__init__()
        #   query matrix
        self.q = nn.Linear(input_planes, input_planes)
        #   key matrix
        self.k = nn.Linear(input_planes, input_planes)
        #   value matrix
        self.v = nn.Linear(input_planes, input_planes)

        self.dropout_attention = nn.Dropout(0.1)

    def forward(self, x):
        batch, d_model, seq_len, time_len = x.size()
        x = x.transpose(3, 1)

        q, k, v = self.q(x), self.k(x), self.v(x)
        q = q.view(batch, time_len, seq_len, d_model)
        k = k.view(batch, time_len, seq_len, d_model)
        v = v.view(batch, time_len, seq_len, d_model)

        logits = torch.matmul(q, k.transpose(-2, -1))
        dk = m.sqrt(d_model + 1e-7)
        scaled_logits = torch.div(logits, dk)
        weights = nn.functional.softmax(scaled_logits, dim=-1)
        weights = self.dropout_attention(weights)
        attention_output = torch.matmul(weights, v)
        return attention_output

class ECNN_BASE_CM_SA(nn.Module):
    def __init__(self, num_classes, input_width, input_height, kernel_size, dilations, planes, attention_mode, attention_hint):
        super(ECNN_BASE_CM_SA, self).__init__()
        self.attention_mode = attention_mode
        self.planes = planes
        self.attention_hint = attention_hint

        self.conv1 = nn.Conv2d(3, planes, kernel_size=(1, kernel_size), stride=(1, 2),
                               padding=(0,  0))
        self.temp = int((input_width - kernel_size)/2) + 1

        self.paddings = [(0, 0), (0, 15), (0, 25)]
        self.kernel_size_new = [85, 115, 135]

        self.conv2_l1 = nn.Conv2d(planes, 2*planes, kernel_size=(1, self.kernel_size_new[0]),
                                  stride=(1, 2), padding=self.paddings[0], dilation=(1, dilations[0]))
        self.conv2_l2 = nn.Conv2d(planes, 2*planes, kernel_size=(1, self.kernel_size_new[1]),
                                  stride=(1, 2), padding=self.paddings[1], dilation=(1, dilations[1]))
        self.conv2_l3 = nn.Conv2d(planes, 2*planes, kernel_size=(1, self.kernel_size_new[2]),
                                  stride=(1, 2), padding=self.paddings[2], dilation=(1, dilations[2]))
        self.maxpool2 = nn.MaxPool2d((1, 2), stride=(1, 2))

        self.temp = int((self.temp - self.kernel_size_new[0])/2) + 1
        self.conv3 = nn.Conv2d(3*2*planes, 3*2*planes, kernel_size=(1, self.temp), stride=(1, 1),
                               padding=(0, 0))
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(3*1*input_height*2*planes, 4*planes)
        self.fc2 = nn.Linear(4*planes, num_classes)

        if(self.attention_mode):
            self.attention = SelfAttention(3*2*planes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        temp = self.conv2_l1(x)
        multiscale_x_1 = self.relu(temp)
        temp = self.conv2_l2(x)
        multiscale_x_2 = self.relu(temp)
        temp = self.conv2_l3(x)
        multiscale_x_3 = self.relu(temp)
        x = torch.cat((multiscale_x_1, multiscale_x_2, multiscale_x_3), dim=1)

        x = self.conv3(x)
        x = self.relu(x)
        if(self.attention_mode):
            atten_x = self.attention(x)
            atten_x = atten_x.transpose(1, 3)
            x = atten_x + x
            x = self.relu(x)
        else:
            pass
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

def ECNN_BASE_CM_SA_entry(num_classes, input_width, input_height, kernel_size, dilations, planes, attention_mode, attention_hint):
    model = ECNN_BASE_CM_SA(num_classes, input_width, input_height, kernel_size, dilations, planes, attention_mode, attention_hint)
    return model

class ECNN_BASE_PCM_SA(nn.Module):
    def __init__(self, num_classes, input_width, input_height, kernel_size, dilations, planes, attention_mode, attention_hint):
        super(ECNN_BASE_PCM_SA, self).__init__()
        self.attention_mode = attention_mode
        self.planes = planes
        self.attention_hint = attention_hint

        self.conv1 = nn.Conv2d(3, planes, kernel_size=(1, kernel_size), stride=(1, 2),
                               padding=(0,  0))
        self.temp = int((input_width - kernel_size)/2) + 1

        self.paddings = [(0, 0), (0, 0), (0, 0)]
        self.kernel_size_new = [85, 43, 29]
        self.conv2_l1 = nn.Conv2d(planes, 2*planes, kernel_size=(1, self.kernel_size_new[0]),
                                  stride=(1, 2), padding=self.paddings[0], dilation=(1, dilations[0]))
        self.conv2_l2 = nn.Conv2d(planes, 2*planes, kernel_size=(1, self.kernel_size_new[1]),
                                  stride=(1, 2), padding=self.paddings[1], dilation=(1, dilations[1]))
        self.conv2_l3 = nn.Conv2d(planes, 2*planes, kernel_size=(1, self.kernel_size_new[2]),
                                  stride=(1, 2), padding=self.paddings[2], dilation=(1, dilations[2]))
        self.maxpool2 = nn.MaxPool2d((1, 2), stride=(1, 2))
        self.temp = int((self.temp - self.kernel_size_new[0])/2) + 1

        self.conv3 = nn.Conv2d(2*planes, 2*planes, kernel_size=(1, self.temp), stride=(1, 1),
                               padding=(0, 0))

        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(1*input_height*2*planes, 4*planes)
        self.fc2 = nn.Linear(4*planes, num_classes)

        if(self.attention_mode):
            self.attention = SelfAttention(2*planes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        temp = self.conv2_l1(x)
        multidilation_x_1 = self.relu(temp)
        temp = self.conv2_l2(x)
        multidilation_x_2 = self.relu(temp)
        multidilation_x_1 = multidilation_x_1 + multidilation_x_2
        temp = self.conv2_l3(x)
        multidilation_x_3 = self.relu(temp)
        multidilation_x_1 = multidilation_x_1 + multidilation_x_3
        x = multidilation_x_1

        x = self.conv3(x)
        x = self.relu(x)
        if(self.attention_mode):
            atten_x = self.attention(x)
            atten_x = atten_x.transpose(1, 3)
            x = atten_x + x
            x = self.relu(x)
        else:
            pass
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

def ECNN_BASE_PCM_SA_entry(num_classes, input_width, input_height, kernel_size, dilations, planes, attention_mode, attention_hint):
    model = ECNN_BASE_PCM_SA(num_classes, input_width, input_height, kernel_size, dilations, planes, attention_mode, attention_hint)
    return model


class Mixed_SelfAttention(nn.Module):
    def __init__(self, input_planes):
        super(Mixed_SelfAttention, self).__init__()
        #   query matrix
        self.q = nn.Linear(input_planes, input_planes)
        #   key matrix
        self.k = nn.Linear(input_planes, input_planes)
        #   value matrix
        self.v = nn.Linear(input_planes, input_planes)

        self.dropout_attention = nn.Dropout(0.1)

    def forward(self, x, x_copula):
        batch, d_model, seq_len, time_len = x.size()
        x = x.transpose(3, 1)

        q, k, v = self.q(x), self.k(x), self.v(x)
        q = q.view(batch, time_len, seq_len, d_model)
        k = k.view(batch, time_len, seq_len, d_model)
        v = v.view(batch, time_len, seq_len, d_model)

        logits = torch.matmul(q, k.transpose(-2, -1))
        original_logits = logits
        '''modify attention matrix'''
        logits = torch.mul(logits, x_copula)
        dk = m.sqrt(d_model + 1e-7)
        scaled_logits = torch.div(logits, dk)
        weights = nn.functional.softmax(scaled_logits, dim=-1)
        matrix_weights = weights
        weights = self.dropout_attention(weights)
        attention_output = torch.matmul(weights, v)
        return attention_output, matrix_weights, original_logits

class MTSD(nn.Module):
    def __init__(self, num_classes, input_width, input_height, kernel_size, dilations, planes, attention_mode, attention_hint):
        super(MTSD, self).__init__()
        self.attention_mode = attention_mode
        self.planes = planes
        self.attention_hint = attention_hint

        self.conv1 = nn.Conv2d(3, planes, kernel_size=(1, kernel_size), stride=(1, 2),
                               padding=(0,  0))
        self.temp = int((input_width - kernel_size)/2) + 1

        self.paddings = [(0, 0), (0, 0), (0, 0)]
        self.kernel_size_new = [85, 43, 29]

        self.conv2_l1 = nn.Conv2d(planes, 2*planes, kernel_size=(1, self.kernel_size_new[0]),
                                  stride=(1, 2), padding=self.paddings[0], dilation=(1, dilations[0]))
        self.conv2_l2 = nn.Conv2d(planes, 2*planes, kernel_size=(1, self.kernel_size_new[1]),
                                  stride=(1, 2), padding=self.paddings[1], dilation=(1, dilations[1]))
        self.conv2_l3 = nn.Conv2d(planes, 2*planes, kernel_size=(1, self.kernel_size_new[2]),
                                  stride=(1, 2), padding=self.paddings[2], dilation=(1, dilations[2]))
        self.temp = int((self.temp - self.kernel_size_new[0])/2) + 1
        print(self.temp)
        # self.conv3 = nn.Conv2d(2*planes, 2*planes, kernel_size=(1, self.temp), stride=(1, 1),
        #                        padding=(0, 0))
        self.conv3 = nn.Conv2d(2 * planes, 2 * planes, kernel_size=(1, self.temp*3), stride=(1, 1),
                                                      padding=(0, 0))
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(1*input_height*2*planes, 4*planes)
        self.fc2 = nn.Linear(4*planes, num_classes)

        #   attention part
        if(self.attention_mode):
            self.attention = Mixed_SelfAttention(2*planes)


    def forward(self, x):
    # def forward(self, x, x_copula):
        x_copula = x.clone()
        x_copula = x_copula[:, 0, :, 0:62]
        print('x_copula size {}'.format(x_copula.size()))
        x = self.conv1(x)
        x = self.relu(x)
        temp = self.conv2_l1(x)
        multidilation_x_1 = self.relu(temp)
        temp = self.conv2_l2(x)
        multidilation_x_2 = self.relu(temp)
        # multidilation_x_1 = multidilation_x_1 + multidilation_x_2
        temp = self.conv2_l3(x)
        multidilation_x_3 = self.relu(temp)
        # multidilation_x_1 = multidilation_x_1 + multidilation_x_3
        # x = multidilation_x_1
        x = torch.cat((multidilation_x_1, multidilation_x_2, multidilation_x_3), dim=3)
        print(x.size())
        x = self.conv3(x)
        x = self.relu(x)
        print('after conv3 {}'.format(x.size()))


        if(self.attention_mode):
            print('x_copula size {}'.format(x_copula.size()))
            print('x size {}'.format(x.size()))
            atten_x, self.matrix_atten , self.original_atten = self.attention(x, x_copula)
            atten_x = atten_x.transpose(1, 3)
            x = atten_x + x
            x = self.relu(x)
        else:
            pass
        
        print('after atten {}'.format(x.size()))
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

def MTSD_entry(num_classes, input_width, input_height, kernel_size, dilations, planes, attention_mode, attention_hint):
    model = MTSD(num_classes, input_width, input_height, kernel_size, dilations, planes, attention_mode, attention_hint)
    return model

