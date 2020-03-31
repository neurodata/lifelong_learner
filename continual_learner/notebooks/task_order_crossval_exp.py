#%%
import matplotlib.pyplot as plt
import random
import pickle
from skimage.transform import rotate
from scipy import ndimage
from skimage.util import img_as_ubyte
from joblib import Parallel, delayed
import numpy as np
from itertools import product
import torch
import os
# %%
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def data_shape_continual_learn(X):
    mat = np.zeros((4,32,32),dtype=int)
    tmp = X.reshape(3,32,32)
    mat[0:3,:,:] = tmp
    mat = mat.reshape(64,64)
    
    return mat

def homogenize_labels(a):
    u = np.unique(a)
    return np.array([np.where(u == i)[0][0] for i in a])


def dehomogenize(labels, task=1):
    u = np.unique(labels)
    return np.array([(task-1)*10+np.where(u == i)[0][0] for i in labels])

#%%
def cross_val_data(data_x, data_y, class_idx, total_cls=100, cv=1):
    x = data_x.copy()
    y = data_y.copy()
    idx = class_idx.copy()
    
    
    for i in range(total_cls):
        indx = np.roll(idx[i],(cv-1)*100)
        
        if i==0:
            train_x = x[indx[0:500],:,:]
            test_x = x[indx[500:600],:,:]
            train_y = y[indx[0:500]]
            test_y = y[indx[500:600]]
        else:
            train_x = np.concatenate((train_x, x[indx[0:500],:,:]), axis=0)
            test_x = np.concatenate((test_x, x[indx[500:600],:,:]), axis=0)
            train_y = np.concatenate((train_y, y[indx[0:500]]), axis=0)
            test_y = np.concatenate((test_y, y[indx[500:600]]), axis=0)
        
        
    train_data = (torch.tensor(train_x),torch.tensor(np.ravel(train_y)))
    test_data = (torch.tensor(test_x),torch.tensor(np.ravel(test_y)))
        
    return train_data, test_data

#%%
def run_nn(num):
    if num == 0:
        get_ipython().system("../main.py --experiment=splitMNIST --tasks=10 --scenario=task --si --c=0.1 --iters=2000 --name 'tmp_syn1' --tasks_to_complete=10")
    elif num == 1:
        get_ipython().system("../main.py --experiment=splitMNIST --tasks=10 --scenario=task --replay=current --distill --iters=2000 --name 'tmp_syn1' --tasks_to_complete=10")
    elif num == 2:
        get_ipython().system("../main.py --experiment=splitMNIST --tasks=10 --scenario=task --ewc --lambda=5000 --iters=2000 --name 'tmp_syn1' --tasks_to_complete=10")
    elif num == 3:
        get_ipython().system("../main.py --experiment=splitMNIST --tasks=10 --scenario=task --ewc --online --lambda=5000 --gamma=1 --iters=2000 --name 'tmp_syn1' --tasks_to_complete=10")


#%%
def change_labels(labels):
    lbl = labels.copy()
    l = len(lbl)
    for i in range(l):
        lbl[i] = np.mod(lbl[i]+10,100)

    return lbl
#%%
train_file = '../train'
unpickled_train = unpickle(train_file)
train_keys = list(unpickled_train.keys())
fine_labels = np.array(unpickled_train[train_keys[2]])
labels = fine_labels

test_file = '../test'
unpickled_test = unpickle(test_file)
test_keys = list(unpickled_test.keys())
fine_labels = np.array(unpickled_test[test_keys[2]])
labels_ = fine_labels

#%%
total_data = len(labels) 
train_data = np.zeros((total_data,4,32,32),dtype=int)

for i in range(total_data):
    tmp = unpickled_train[b'data'][i].reshape(3,32,32)
    train_data[i,0:3,:,:] = tmp

    
train_data = train_data.reshape(total_data,64,64)

total_data = len(labels_) 
test_data = np.zeros((total_data,4,32,32),dtype=int)

for i in range(total_data):
    tmp = unpickled_test[b'data'][i].reshape(3,32,32)
    test_data[i,0:3,:,:] = tmp

    
test_data = test_data.reshape(total_data,64,64)

data_x = np.zeros((60000,64,64),dtype=int)
data_x[0:50000,:,:] = train_data
data_x[50000:60000,:] = test_data

data_y = np.zeros((60000,),dtype=int)
data_y[0:50000] = labels
data_y[50000:60000] = labels_

#%%
reps = 10
alg = ['SI','LwF','EWC','Online_EWC']
filename = '../order_res'

if not os.path.exists(filename):
    os.mkdir(filename)

for i in range(reps):

    class_idx_train = [np.where(labels== u)[0] for u in np.unique(labels)]
    class_idx_test = [np.where(labels_== u)[0] for u in np.unique(labels_)]
    
    train_grp = (torch.tensor(train_data),torch.tensor(np.ravel(labels)))
    test_grp = (torch.tensor(test_data),torch.tensor(np.ravel(labels_)))
    class_idx = [np.where(data_y == u)[0] for u in np.unique(data_y)]

    for cv in range(6):
        train_grp, test_grp = cross_val_data(data_x,data_y,class_idx,cv=cv+1)
        torch.save(train_grp,'../datasets/mnist/MNIST/processed/training.pt')
        torch.save(test_grp,'../datasets/mnist/MNIST/processed/test.pt')

        for ii in range(4):
            run_nn(ii)
            err = np.zeros((10,10),dtype=float)

            for jj in range(10):
                err[jj,:] = unpickle('../tmp_syn1/task'+str(jj+1)+'.pickle')

            with open(filename+'/'+alg[ii]+str(i+1)+'_'+str(cv+1),'wb') as f:
                pickle.dump(err,f)

    data_y = change_labels(data_y)

get_ipython().system("sudo shutdown now")



# %%
