# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 17:17:11 2022

@author: hp
"""

import glob 
import json
import csv

#preparing output file SeverePain
fea = 468
base = 2

fpose = open('C:\\project\\STALSTM\\dataset\\KKH_1to48_preprocessed' + str(base) +'keypoints.csv', 'w', newline='\n')
writer = csv.writer(fpose)

# reading input file
for name in glob.glob('C:\\project\\STALSTM\\dataset\\KKH_1to48_preprocessed\\Severe pain\\*.txt'):
    #print(name) 
    line_count = 0
    l = []
    with open(name) as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')

      for row in csv_reader:
        if line_count == 0:
            #print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            nID = int(row[0])
            if (nID%base)==0:
                #modulo in pyhton
                l.append(row[2])
                l.append(row[3])
                line_count += 1
            if int(row[1])== 467:
                if int(row[0])== 60:
                    l.append(1)
                    l.append(0)
                    l.append(0)
                    if len(l) == (936 * 30 + 3):
                        writer.writerow(l)
                    l = []
    if (line_count<(fea*base)):
        print(name + " having only " + str(line_count) + " dataset")

print(name + " having " + str(line_count) + " dataset")



# reading input file
for name in glob.glob('C:\\project\\STALSTM\\dataset\\KKH_1to48_preprocessed\\Mild pain\\*.txt'):
    #print(name)
    line_count = 0
    l = []
    with open(name) as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')

      for row in csv_reader:
        if line_count == 0:
            #print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            nID = int(row[0])
            if (nID%base)==0:
                #modulo in pyhton
                l.append(row[2])
                l.append(row[3])
                line_count += 1
            if int(row[1])== 467:
                if int(row[0])== 60:
                    l.append(0)
                    l.append(1)
                    l.append(0)

                    if len(l) == (936 * 30 + 3):
                      writer.writerow(l)

                    l = []

    if (line_count<(fea*base)):
        print(name + " having only " + str(line_count) + " dataset")

print(name + " having " + str(line_count) + " dataset")

# reading input file
for name in glob.glob('C:\\project\\STALSTM\\dataset\\KKH_1to48_preprocessed\\No pain\\*.txt'):
    #print(name)
    line_count = 0
    l = []
    with open(name) as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')

      for row in csv_reader:
        if line_count == 0:
            #print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            nID = int(row[0])
            if (nID%base)==0:
                #modulo in pyhton
                l.append(row[2])
                l.append(row[3])
                line_count += 1
            if int(row[1])== 467:
                if int(row[0])== 60:
                    l.append(0)
                    l.append(0)
                    l.append(1)
                    if len(l) == (936 * 30 + 3):
                        writer.writerow(l)
                    l = []

    if (line_count<(fea*base)):
        print(name + " having only " + str(line_count) + " dataset")

print(name + " having " + str(line_count) + " dataset")
########
###test dataset
########

# # reading input file
# for name in glob.glob('C:\\project\\PainData\\preprocessed\\Severe_test\\*.txt'):
#     #print(name)
#     line_count = 0
#     l = []
#     with open(name) as csv_file:
#       csv_reader = csv.reader(csv_file, delimiter=',')
#
#       for row in csv_reader:
#         if line_count == 0:
#             #print(f'Column names are {", ".join(row)}')
#             line_count += 1
#         else:
#             nID = int(row[0])
#             if (nID%base)==0:
#                 #modulo in pyhton
#                 l.append(row[2])
#                 l.append(row[3])
#                 line_count += 1
#             if int(row[1])== 467:
#                 if int(row[0])== 60:
#                     l.append(1)
#                     l.append(0)
#                     l.append(0)
#                     writer.writerow(l)
#                     l = []
#
#     if (line_count<(fea*base)):
#         print(name + " having only " + str(line_count) + " dataset")
#
# print(name + " having " + str(line_count) + " dataset")
#
# # reading input file
# for name in glob.glob('C:\\project\\PainData\\preprocessed\\Mild_test\\*.txt'):
#     #print(name)
#     line_count = 0
#     l = []
#     with open(name) as csv_file:
#       csv_reader = csv.reader(csv_file, delimiter=',')
#
#       for row in csv_reader:
#         if line_count == 0:
#             #print(f'Column names are {", ".join(row)}')
#             line_count += 1
#         else:
#             nID = int(row[0])
#             if (nID%base)==0:
#                 #modulo in pyhton
#                 l.append(row[2])
#                 l.append(row[3])
#                 line_count += 1
#             if int(row[1])== 467:
#                 if int(row[0])== 60:
#                     l.append(0)
#                     l.append(1)
#                     l.append(0)
#                     writer.writerow(l)
#                     l = []
#
#     if (line_count<(fea*base)):
#         print(name + " having only " + str(line_count) + " dataset")
#
# print(name + " having " + str(line_count) + " dataset")
#
# # reading input file
# for name in glob.glob('C:\\project\\PainData\\preprocessed\\No_test\\*.txt'):
#     #print(name)
#     line_count = 0
#     l = []
#     with open(name) as csv_file:
#       csv_reader = csv.reader(csv_file, delimiter=',')
#
#       for row in csv_reader:
#         if line_count == 0:
#             #print(f'Column names are {", ".join(row)}')
#             line_count += 1
#         else:
#             nID = int(row[0])
#             if (nID%base)==0:
#                 #modulo in pyhton
#                 l.append(row[2])
#                 l.append(row[3])
#                 line_count += 1
#             if int(row[1])== 467:
#                 if int(row[0])== 60:
#                     l.append(0)
#                     l.append(0)
#                     l.append(1)
#                     writer.writerow(l)
#                     l = []
#
#     if (line_count<(fea*base)):
#         print(name + " having only " + str(line_count) + " dataset")

print(name + " having " + str(line_count) + " dataset")

fpose.close()