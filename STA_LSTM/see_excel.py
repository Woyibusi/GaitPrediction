## Use this to see excel file in pycharm
import pandas as pd

table1 = pd.read_csv("C:\project\STALSTM\dataset\All raw clips_keypoints_26to100_SG_preprocessed2.csv")
print(table1.head())

table = pd.read_csv("C:\project\STALSTM\dataset\old-All raw clips_keypoints_26to100_SG_preprocessed2.csv")
print(table.head())

## Debug to view table and table1