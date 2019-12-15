import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
import glob

from qore_sdk.utils import sliding_window
from qore_sdk.featurizer import Featurizer
from qore_sdk.client import WebQoreClient

from pydub import AudioSegment
from scipy import signal

def SplitTune(dataset,dec,width = 44000):
    print('Width:',width)
    dataset_separate = []
    for n,d in enumerate(dataset):
        start = int(d.shape[0]*0.01)
        end = int(d.shape[0]*0.99)
        num = int((end-start)/width)
        print(start,end,num)
        for i in range(num):
            hoge = d[(start+(i*width)):(start+(i*width))+width]
            if(len(hoge) == width):
                dataset_separate.append(hoge)
            else:
                pass
    dataset_separate = np.array(dataset_separate)
    answer = np.zeros(dataset_separate.shape[0])+dec
    print('Shape of separated dataset:',dataset_separate.shape)
    return(dataset_separate,answer)

def PrepareTrain(dataset_separate,target1,target2):
    X_train = np.vstack((dataset_separate[int(target1)],dataset_separate[int(target2)]))
    y_train = np.zeros(X_train.shape[0])
    y_train[int(y_train.shape[0]/2):] = 1
    print('X_train shape:',X_train.shape)
    print('y_train,shape:',y_train.shape)
    return(X_train, y_train)

def PermutateTrain(X_train,y_train,lim=None):
    idx = np.arange(0,X_train.shape[0],1)
    idx = np.random.permutation(idx)
    idx_lim = idx[:lim]
    X_train = X_train[idx_lim]
    y_train = y_train[idx_lim]
    return X_train,y_train,idx_lim