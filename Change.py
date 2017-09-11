import numpy as np

def changetpye1(y_train):
    #print y_train
    #y_train.sort_index()
    yy = np.zeros((891, 9))
    #y_train=y_train.as_matrix()
    y2 = np.array(y_train)
    y3 = y2.reshape(891, 1)
    for i in range(len(y3)):
        yy[i][y3[i][0] - 1] = 1
        print y3[i][0]
        print  yy[i]

    #print y3

    return yy

def changetpye2(y_train):
   # y_train.sort_index()
    yy = np.zeros((9, 9))
    #y_train = y_train.as_matrix()
    y2 = np.array(y_train)
    y3 = y2.reshape(9, 1)
    for i in range(len(y3)):
        yy[i][y3[i][0] - 1] = 1
    return yy

def changetpye3(y_train):
    #y_train=y_train.sort_index()
    #print y_train
    labels=y_train.as_matrix()
    labels=np.array(labels).reshape(-1)
    labels=np.eye(10)[labels]

    return labels