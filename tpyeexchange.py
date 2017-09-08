
import numpy as np

def changetpye1(y_train):
    #print y_train
    #y_train.sort_index()
    yy = np.zeros((540, 9))
    y2 = np.array(y_train)
    y3 = y2.reshape(540, 1)
    for i in range(len(y3)):
        yy[i][y3[i][0] - 1] = 1

    #print y3

    return yy

def changetpye2(y_train):
    y_train.sort_index()
    yy = np.zeros((360, 9))
    y2 = np.array(y_train)
    y3 = y2.reshape(360, 1)
    for i in range(len(y3)):
        yy[i][y3[i][0] - 1] = 1
    return yy