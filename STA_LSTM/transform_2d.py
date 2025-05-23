import numpy as np
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