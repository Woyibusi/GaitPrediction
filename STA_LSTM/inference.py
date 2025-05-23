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

#from modelbase import LSTM as Net

# from modelbase import SA_LSTM as Net
# from modelbase import TA_LSTM as Net
# from modelbase import LSTM as Net
# from modelbase import FCN as Net
# from modelbase import SVM as Net

'''****************************initialization*******************************''' 
IN_DIM =  1404 
SEQUENCE_LENGTH = 1404  

LSTM_IN_DIM = int(IN_DIM/SEQUENCE_LENGTH)     
LSTM_HIDDEN_DIM = 18  

OUT_DIM = 3   

LEARNING_RATE = 0.05 # learning rate
WEIGHT_DECAY = 1e-6   

BATCH_SIZE = 256       

EPOCHES = 100  

TRAIN_PER = 0.01 
VALI_PER = 0.0 


#USE_GPU = torch.cuda.is_available()
#print(USE_GPU)
USE_GPU = False

   

'''****************************data prepration*******************************''' 
#print('Loading data')
dp = data_preprocess(file_path = 'C:\\project\\STALSTM\\data\\dataset\\raw_data_nopain.csv', train_per = TRAIN_PER, vali_per = VALI_PER, in_dim = IN_DIM)

#print('Loading data...')
raw_data = dp.load_data()

#print('Data loaded')

(train_data,train_groundtruth),(vali_data,vali_groundtruth),(test_data,test_groundtruth) = dp.split_data(raw_data = raw_data, _type = 'linear')


# 设置对数据进行的转换方式，transform.compose的作用是将多个transform组合到一起进行使用
transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=(0,0,0),std=(1,1,1))])
# print('数据转换为tensor')

# data_trans返回的值是一个字典，内部包含数据和真值{'inputs':inputs,'groundtruth':groundtruths}

# 准备训练集
train_data_trans = data_trans(train_data,train_groundtruth,transform)

train_dataloader = torch.utils.data.DataLoader(train_data_trans,
                                           batch_size =BATCH_SIZE,
                                           shuffle = True,
                                           num_workers = 4)

test_data_trans = data_trans(test_data, test_groundtruth,transform)

test_dataloader = torch.utils.data.DataLoader(test_data_trans,
                                           batch_size = BATCH_SIZE,
                                           shuffle = False,
                                           num_workers = 4)



'''****************************model prepration*******************************''' 
net = torch.load('C:\\project\\STALSTM\\models\\sta_lstm_36.pth')
error_criterion = nn.L1Loss() # MSELoss()
root = os.getcwd()
print(root)


def test():
    
    error = 0.0
    predictions = []
    test_groundtruths = []

    # 告诉网络进行测试，不再是训练模式
    net.eval() 

    for i,data in enumerate(test_dataloader):

        inputs = data['inputs']
        groundtruths = data['groundtruths']     
        #print(len(input))    
        if USE_GPU:

            inputs = Variable(inputs).cuda()
            groundtruths = Variable(groundtruths).cuda()
            
        else:
            
            inputs = Variable(inputs)
            groundtruths = Variable(groundtruths)

        
        out = net(inputs)
        #print(out.shape)
        err = error_criterion(out,groundtruths)
        #print('Error 1 = ', err)
        #error += (error_criterion(out,groundtruths).item()*groundtruths.size(0))
        #print('Error 2 = ', error)
        
        if USE_GPU:
            predictions.extend(out.cpu().data.numpy().tolist())
            test_groundtruths.extend(groundtruths.cpu().data.numpy().tolist())
            
        else:
            predictions.extend(out.data.numpy().tolist())
            test_groundtruths.extend(groundtruths.data.numpy().tolist())
    
        
    #print(len(test_data_trans))
    #average_error = np.sqrt(error/len(test_data_trans))
    
    #return np.array(predictions).reshape((len(predictions))),np.array(test_groundtruths).reshape((len(test_groundtruths))),average_error
    #out = out.detach().numpy()
    #groundtruths = groundtruths.detach().numpy()
      
    correct_pred = tf.equal(tf.argmax(predictions,1), tf.argmax(test_groundtruths,1))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    print(confusion_matrix(tf.argmax(test_groundtruths,1), tf.argmax(predictions,1)))
        
        
    return acc, err

def main():

    
    test_start = time.time()
    print(test_start)
    
    acc, average_error = test()

    acc = acc * 100.0
    
    #print(calculate_accuracy(test_groundtruth, predictions))
    print('test time = {}s'.format(int((time.time() - test_start)+1.0)))
    print('Loss = ',  average_error)
    print('Test accuracy is =', acc)

    #result = pd.DataFrame(data = {'Q(t+1)':predictions,'Q(t+1)truth':test_groundtruth})
    #result.to_csv('C:\\project\\STALSTM\\out_t+1.csv')
    


if __name__ == '__main__':
    main()

