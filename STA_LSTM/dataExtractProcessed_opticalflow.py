# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 17:17:11 2022

@author: hp
"""

import glob 
import json
import csv
import pandas as pd
import os
import numpy as np

#preparing output file SeverePain
fea = 468
base = 2

fpose = open('C:\\project\\STALSTM\\dataset\\combined data upto 13Nov2022_preprocessed' + str(base) +'.csv', 'w', newline='\n')
writer = csv.writer(fpose)

# # reading input file
for name in glob.glob('C:\\project\\STALSTM\\dataset\\combined data upto 13Nov2022_preprocessed\\Severe pain\\*.txt'):
    # print(name)
    data = pd.read_csv(name, sep =',', header = 0)      # read txt file as pandas dataframe
    data_b2 = data.drop(data[data.frame % base ==0].index).to_numpy()  # remove frames
    data_trans = data_b2.T

    x = data_trans[2]
    y = data_trans[3]
    x_diff = [x[i+fea]-x[i] for i in range(len(x)-fea)]
    y_diff = [y[i+fea]-y[i] for i in range(len(y)-fea)]

    l = []
    for x1, y1 in zip(x_diff, y_diff):
        l.append(x1 )
        l.append(y1)
    l.append(1)     # Label column
    l.append(0)     # Label column
    l.append(0)     # Label column
    if not np.isnan(l).any():
        if len(l) == (936*29+3):
            writer.writerow(l)
    # print(l)
    # l = []
print("Read all files in folder: {}".format(os.path.dirname(name)))

# reading input file
for name in glob.glob('C:\\project\\STALSTM\\dataset\\combined data upto 13Nov2022_preprocessed\\Mild pain\\*.txt'):
    # print(name)
    data = pd.read_csv(name, sep=',', header=0)
    data_b2 = data.drop(data[data.frame % base == 0].index).to_numpy()
    data_trans = data_b2.T

    x = data_trans[2]
    y = data_trans[3]
    x_diff = [x[i + fea] - x[i] for i in range(len(x) - fea)]
    y_diff = [y[i + fea] - y[i] for i in range(len(y) - fea)]

    l = []
    for x1, y1 in zip(x_diff, y_diff):
        l.append(x1 )
        l.append(y1 )
    l.append(0)  # Label column
    l.append(1)  # Label column
    l.append(0)  # Label column
    if not np.isnan(l).any():
        if len(l) == (936 * 29 + 3):
            writer.writerow(l)
    # print(l)
    # l = []
print("Read all files in folder: {}".format(os.path.dirname(name)))

# reading input file
for name in glob.glob('C:\\project\\STALSTM\\dataset\\combined data upto 13Nov2022_preprocessed\\No pain\\*.txt'):
    # print(name)
    data = pd.read_csv(name, sep=',', header=0)
    data_b2 = data.drop(data[data.frame % base == 0].index).to_numpy()
    data_trans = data_b2.T

    x = data_trans[2]
    y = data_trans[3]
    x_diff = [x[i + fea] - x[i] for i in range(len(x) - fea)]
    y_diff = [y[i + fea] - y[i] for i in range(len(y) - fea)]

    l = []
    for x1, y1 in zip(x_diff, y_diff):
        l.append(x1)
        l.append(y1)
    l.append(0)  # Label column
    l.append(0)  # Label column
    l.append(1)  # Label column
    if not np.isnan(l).any():
        if len(l) == (936 * 29 + 3):
            writer.writerow(l)
    # print(l)
    # l = []
print("Read all files in folder: {}".format(os.path.dirname(name)))

fpose.close()
