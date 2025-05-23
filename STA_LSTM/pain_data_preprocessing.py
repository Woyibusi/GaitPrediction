import os
import numpy as np
## path containing keypoints data
path='C:/Users\dutie\OneDrive\Documents\pain detection\combined data upto 13Nov2022'
path_new='C:/Users\dutie\OneDrive\Documents\pain detection\pain_data_preprocessed'
os.chdir(path)  ## change directory
def est_similarity_trans(x1,y1,x2,y2):
    #x1,y1,x2,y2 are the corrdinates of facial keypoints 10 and 152 (using mediapipe)
    x1p,y1p,x2p,y2p=0.5,0.4,0.5,0.6 # new corrdiantes after similarity transformation within (0,1)
    A=np.array([[x1,-y1,1,0],[y1,x1,0,1],[x2,-y2,1,0],[y2,x2,0,1]])
    b=np.array([[x1p],[y1p],[x2p],[y2p]])
    T_matrix=np.dot(np.linalg.inv(np.dot(np.transpose(A),A)),np.dot(np.transpose(A),b))
    return T_matrix

def similarity_trans(x,y,T):
    A = np.array([[x, -y, 1, 0], [y, x, 0, 1]])
    b=np.dot(A,T)
    #print(np.transpose(b))
    b1=np.dot(np.transpose(b),[[2160,0],[0,3840]])
    return b1
#T=est_similarity_trans(573,512,546,791)
#similarity_trans(546,791,T)

temp=[]
for entry in os.listdir(path):              ## loop in the folder
    #max_x,max_y=0,0
    if os.path.isdir(os.path.join(path, entry)):        ## if it is a subfolder
        sequence=0
        for file in os.listdir(os.path.join(path, entry)):      ## loop through all files
            subpath=os.path.join(path, entry)

            if os.path.isfile(os.path.join(subpath, file)):
                if file.endswith('faciallandmarks.txt'):
                    sequence+=1
                    #os.makedirs(os.path.join(subpath,str(sequence)))
                    full_filename=os.path.join(subpath, file)
                    text=open(full_filename)
                    pre_frame=-1
                    for line in text:
                        #find keypoint 10 &152 from first frame
                        if not line.startswith('frame'):
                            frame,id,x,y = line.split(',')
                            if int(id)==10:
                                x1,y1=int(x),int(y)
                            if int(id)==152:
                               x2,y2=int(x),int(y)
                               T=est_similarity_trans(x1, y1, x2, y2)
                               break

                    subpath1 = os.path.join(path_new, entry)
                    full_filename1 = os.path.join(subpath1, file)
                    text1 = open(full_filename)
                    with open(full_filename1, 'w') as f:
                        for line in text1:
                            if not line.startswith('frame'):
                                frame, id, x, y = line.split(',')
                                new_xy=similarity_trans(int(x), int(y), T)
                                f.write(frame+','+id+','+str(int(new_xy[0][0]))+','+str(int(new_xy[0][1])))
                                f.write('\n')
                            else:
                                f.write("frame, id, x, y")
                                f.write('\n')

                        f.close()








