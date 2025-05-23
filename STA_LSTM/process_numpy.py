import os
import numpy as np
from transform3d_scal import est_similarity_trans, similarity_trans
import glob
import json
import csv

# severe pain 100, mild pain 010, no pain 001
#fpose = open(r'C:\projects\STALSTM\dataset\manual combined SGH_KKH\training_data_oct_2024.csv', 'w', newline='\n') # for training
fpose = open(r'C:\projects\STALSTM\dataset\manual combined SGH_KKH\testing_data_oct_2024_no_stride.csv', 'w', newline='\n') # for testing
writer = csv.writer(fpose)
## path containing numpy file of mild pain

## for mild pain
#path = 'C:\projects\STALSTM\dataset\manual combined SGH_KKH\Train test seperate\\train dataset\Mild pain (1-3)'# for training
path = 'C:\projects\STALSTM\dataset\manual combined SGH_KKH\Train test seperate\\test dataset\Mild pain (1-3)'
os.chdir(path)  ## change directory
num_mid = 0
for file in os.listdir(path):  ## loop in the folder
    if os.path.isfile(os.path.join(path, file)):
        if file.endswith('.npy'):
            full_filename = os.path.join(path, file)
            data = np.load(full_filename)
            data_normalized = np.zeros([len(data), 478 * 2])
            for i in range(len(data)):
                source_points = np.array([data[i, 226, :], data[i, 446, :], data[i, 2, :]])
                # print(i)
                # print(source_points)
                R, t ,s= est_similarity_trans(source_points)
                transformed_keypoints = similarity_trans(data[i, :, :], R, t,s)
                only_xy = transformed_keypoints[:, 0:2]  # only take x and y corrdiantes
                data_normalized[i, :] = only_xy.reshape(-1)  # reshape it to 1d array

            ## formate it into 30 frames in a row (478x2x30), stride size 1
            # data_normalized_30frame = np.zeros([len(data_normalized)-29, 478 * 2*30])

            #for j in range(0, len(data_normalized) - 29, 15): # 15 is the striding size to make training samples balance
            for j in range(0, len(data_normalized) - 29, 30):
                l = data_normalized[j:(j + 30), :].reshape(-1)
                l = np.append(l, [0, 1, 0], axis=0)

                num_mid += 1
                #print(num_mid)
                writer.writerow(l)
print('mild pain' + " having " + str(num_mid) + " dataset")

# for severe pain
#path = 'C:\projects\STALSTM\dataset\manual combined SGH_KKH\Train test seperate\\train dataset\Severe pain (4-10)' # for training
path = 'C:\projects\STALSTM\dataset\manual combined SGH_KKH\Train test seperate\\test dataset\Severe pain (4-10)'
os.chdir(path)  ## change directory
num_severe = 0
for file in os.listdir(path):  ## loop in the folder
    if os.path.isfile(os.path.join(path, file)):
        if file.endswith('.npy'):
            full_filename = os.path.join(path, file)
            data = np.load(full_filename)
            data_normalized = np.zeros([len(data), 478 * 2])
            for i in range(len(data)):
                source_points = np.array([data[i, 226, :], data[i, 446, :], data[i, 2, :]])
                # print(i)
                # print(source_points)
                R, t, s = est_similarity_trans(source_points)
                transformed_keypoints = similarity_trans(data[i, :, :], R, t, s)
                only_xy = transformed_keypoints[:, 0:2]  # only take x and y corrdiantes
                data_normalized[i, :] = only_xy.reshape(-1)  # reshape it to 1d array

            ## formate it into 30 frames in a row (478x2x30), stride size 1
            # data_normalized_30frame = np.zeros([len(data_normalized)-29, 478 * 2*30])
            #for j in range(0, len(data_normalized) - 29, 8): # 8 is the striding size to make the training dataset balance
            for j in range(0, len(data_normalized) - 29, 30):
                l = data_normalized[j:(j + 30), :].reshape(-1)
                l = np.append(l, [1, 0, 0], axis=0)

                num_severe += 1
                writer.writerow(l)
print('severe pain' + " having " + str(num_severe) + " dataset")

#path = 'C:\projects\STALSTM\dataset\manual combined SGH_KKH\Train test seperate\\train dataset\ho pain (0)'
path = 'C:\projects\STALSTM\dataset\manual combined SGH_KKH\Train test seperate\\test dataset\ho pain (0)'
os.chdir(path)  ## change directory
num_nopain = 0
for file in os.listdir(path):  ## loop in the folder
    if os.path.isfile(os.path.join(path, file)):
        if file.endswith('.npy'):
            full_filename = os.path.join(path, file)
            data = np.load(full_filename)
            data_normalized = np.zeros([len(data), 478 * 2])
            for i in range(len(data)):
                source_points = np.array([data[i, 226, :], data[i, 446, :], data[i, 2, :]])
                # print(i)
                # print(source_points)
                R, t, s = est_similarity_trans(source_points)
                transformed_keypoints = similarity_trans(data[i, :, :], R, t, s)
                only_xy = transformed_keypoints[:, 0:2]  # only take x and y corrdiantes
                data_normalized[i, :] = only_xy.reshape(-1)  # reshape it to 1d array

            ## formate it into 30 frames in a row (478x2x30), stride size 1
            # data_normalized_30frame = np.zeros([len(data_normalized)-29, 478 * 2*30])
            #for j in range(0, len(data_normalized) - 29, 100):
            for j in range(0, len(data_normalized) - 29, 30):
                l = data_normalized[j:(j + 30), :].reshape(-1)
                l = np.append(l, [0, 0, 1], axis=0)

                num_nopain += 1
                writer.writerow(l)
print('No pain' + " having " + str(num_nopain) + " dataset")
fpose.close()
