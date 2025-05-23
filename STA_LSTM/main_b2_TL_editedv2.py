#!/usr/bin/env python3

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
import tensorflow as tf

from sklearn.metrics import accuracy_score

from data import data_preprocess, data_trans
from modelbase import STA_LSTM as Net

from sklearn.metrics import confusion_matrix

# from modelbase import LSTM as Net

# from modelbase import SA_LSTM as Net
# from modelbase import TA_LSTM as Net
#from modelbase import LSTM as Net
# from modelbase import FCN as Net
# from modelbase import SVM as Net

'''****************************initialization*******************************'''
base = 2
IN_DIM = int(936 * 29)  # always use 60 / base number to get the final datapoints, e.g. 60 / base_2 = 30
SEQUENCE_LENGTH = int(29)  # 2

LSTM_IN_DIM = int(IN_DIM / SEQUENCE_LENGTH)
LSTM_HIDDEN_DIM = 16  # 32 #64

OUT_DIM = 3

LEARNING_RATE = 0.05  # learning rate
#LEARNING_RATE = 0.0005  # learning rate
WEIGHT_DECAY = 1e-6

BATCH_SIZE = 128


EPOCHES = 50

TRAIN_PER = 0.70
VALI_PER = 0.10

# USE_GPU = torch.cuda.is_available()
# print(USE_GPU)
USE_GPU = False

'''****************************data prepration*******************************'''

dp = data_preprocess(file_path='C:\\project\\STALSTM\\dataset\\combined data upto 13Nov2022_preprocessed2.csv',
                     train_per=TRAIN_PER, vali_per=VALI_PER, in_dim=IN_DIM)

raw_data = dp.load_data()

# tt = raw_data
# raw_data = tt
# raw_data = (tt + 327)/ 4200.0

(train_data, train_groundtruth), (vali_data, vali_groundtruth), (test_data, test_groundtruth) = dp.split_data(
    raw_data=raw_data, _type='linear')

# 设置对数据进行的转换方式，transform.compose的作用是将多个transform组合到一起进行使用
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1))])
# print('数据转换为tensor')

# data_trans返回的值是一个字典，内部包含数据和真值{'inputs':inputs,'groundtruth':groundtruths}

# 准备训练集
train_data_trans = data_trans(train_data, train_groundtruth, transform)

train_dataloader = torch.utils.data.DataLoader(train_data_trans,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=4)

test_data_trans = data_trans(test_data, test_groundtruth, transform)

test_dataloader = torch.utils.data.DataLoader(test_data_trans,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=4)

raw_data = []
train_data = []

test_data = []

'''****************************model prepration*******************************'''
net = Net(IN_DIM,SEQUENCE_LENGTH,LSTM_IN_DIM,LSTM_HIDDEN_DIM,OUT_DIM,USE_GPU)
# for transfer learning
# net = torch.load('C:\\project\\STALSTM\\models\\sta_lstm_b2_025_e200.pth')

if USE_GPU:
    net = net.cuda()
    # print('本次实验使用GPU加速')
else:
    pass
    # print('本次实验不使用GPU加速')

# 使用SGD（随机梯度下降）优化，学习率为0.001，动量为0.9
# optimizer = optim.SGD(net.parameters(), lr= LEARNING_RATE, momentum=0.9)
# 根据梯度调整参数数值，Adam算法
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


# 学习率根据训练的次数进行调整
adjust_lr = optim.lr_scheduler.MultiStepLR(optimizer,
                                           milestones=[i * 10 for i in range(EPOCHES // 10)],
                                           gamma=0.5)

# 定义训练损失函数&测试误差函数
# loss_criterion = nn.SmoothL1Loss()
loss_criterion = nn.MSELoss()
error_criterion = nn.L1Loss()  # MSELoss()


def train(verbose=False):
    net.train()
    loss_list = []

    for i, data in enumerate(train_dataloader):

        inputs = data['inputs']
        groundtruths = data['groundtruths']
        y = groundtruths
        if USE_GPU:
            inputs = Variable(inputs).cuda()
            groundtruths = Variable(groundtruths).cuda()

        else:
            inputs = Variable(inputs)
            groundtruths = Variable(groundtruths)

        # 将参数的grad值初始化为0
        optimizer.zero_grad()

        # 获得网络输出结果
        out = net(inputs)

        # 根据真值计算损失函数的值
        loss = loss_criterion(out, groundtruths)

        out = out.detach().numpy()
        groundtruths = groundtruths.detach().numpy()

        correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(groundtruths, 1))
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # print(out.type)
        # print(np.max(y))
        # print(accuracy)

        # print(one_hot(np.array(out)))

        # 通过优化器优化网络
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

    # print(tf.argmax(out,1))
    # print(tf.argmax(groundtruths,1))
    # print(confusion_matrix(tf.argmax(groundtruths,1), tf.argmax(out,1)))

    return loss_list, acc


def test():
    error = 0.0
    predictions = []
    test_groundtruths = []

    # 告诉网络进行测试，不再是训练模式
    net.eval()

    for i, data in enumerate(test_dataloader):

        inputs = data['inputs']
        groundtruths = data['groundtruths']

        if USE_GPU:

            inputs = Variable(inputs).cuda()
            groundtruths = Variable(groundtruths).cuda()

        else:

            inputs = Variable(inputs)
            groundtruths = Variable(groundtruths)

        out = net(inputs)
        err = error_criterion(out, groundtruths)
        # print('Error 1 = ', err)
        # error += (error_criterion(out,groundtruths).item()*groundtruths.size(0))
        # print('Error 2 = ', error)

        if USE_GPU:
            predictions.extend(out.cpu().data.numpy().tolist())
            test_groundtruths.extend(groundtruths.cpu().data.numpy().tolist())

        else:
            predictions.extend(out.data.numpy().tolist())
            test_groundtruths.extend(groundtruths.data.numpy().tolist())

    # print(len(test_data_trans))
    # average_error = np.sqrt(error/len(test_data_trans))

    # return np.array(predictions).reshape((len(predictions))),np.array(test_groundtruths).reshape((len(test_groundtruths))),average_error
    # out = out.detach().numpy()
    # groundtruths = groundtruths.detach().numpy()

    correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(test_groundtruths, 1))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    print(confusion_matrix(tf.argmax(test_groundtruths, 1), tf.argmax(predictions, 1)))

    return acc, err


def main():
    # 记录程序开始的时间
    train_start = time.time()
    loss_recorder = []

    print('starting training... ')

    for epoch in range(EPOCHES):

        loss_list, acc = train(verbose=True)

        loss_recorder.append(np.mean(loss_list))

        acc = acc * 100.0

        print('\nepoch = %d \nloss = %.5f, accuracy = %2.5f' % (epoch + 1, np.mean(loss_list), acc))

        if (epoch % 5 == 0):
            test_start = time.time()
            acc, average_error = test()

            acc = acc * 100.0
            print('Loss = %.5f, Test accuracy is = %2.5f' % (average_error, acc.numpy()))

        # adjust learning rate
        adjust_lr.step()

    print('training time = {}s'.format(int((time.time() - train_start))))

    # 记录测试开始的时间
    test_start = time.time()
    acc, average_error = test()

    acc = acc * 100.0

    # print(calculate_accuracy(test_groundtruth, predictions))
    print('test time = {}s'.format(int((time.time() - test_start) + 1.0)))
    print('Loss = ', average_error)
    print('Test accuracy is =', acc)

    # result = pd.DataFrame(data = {'Q(t+1)':predictions,'Q(t+1)truth':test_groundtruth})
    # result.to_csv('C:\\project\\STALSTM\\out_t+1.csv')

    torch.save(net, 'C:\\project\\STALSTM\\models\\sta_lstm_TL_b2_200_combined_data_upto_13Nov2022_preprocessed2_15Mar.pth')


if __name__ == '__main__':
    main()

