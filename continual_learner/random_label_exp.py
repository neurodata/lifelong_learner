#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import tensorflow as tf
import tensorflow.keras as keras


# In[ ]:


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def data_shape_continual_learn(X):
    mat = np.zeros((4,32,32),dtype=int)
    mat[0:3,:,:] = X
    mat = mat.reshape(64,64)
    
    return mat

def change_label(label, task):
    labels = label.copy()
    lbls_to_change = range(0,10,1)
    lbls_to_transform = range((task-1)*10,task*10,1)

    for count, i in enumerate(lbls_to_change):
        indx = np.where(labels == i)
    
        labels[indx] = -lbls_to_transform[count]
    
    for count, i in enumerate(lbls_to_transform):
        indx = np.where(labels == i)
    
        labels[indx] = lbls_to_change[count]
    
    indx = np.where(labels<0)
    labels[indx] = -labels[indx]
    
    return labels


# In[ ]:


def cross_val_data(data_x, data_y, num_points_per_task, slot_no, total_task=10, shift=1):
    x = data_x.copy()
    y = data_y.copy()
    idx = [np.where(data_y == u)[0] for u in np.unique(data_y)]
    
    sample_per_class = num_points_per_task//total_task

    for task in range(total_task):
        for class_no in range(task*10,(task+1)*10,1):
            indx = np.roll(idx[class_no],(shift-1)*100)
            
            if class_no==0 and task==0:
                train_x = x[indx[slot_no*sample_per_class:(slot_no+1)*sample_per_class],:]
                train_y = y[indx[slot_no*sample_per_class:(slot_no+1)*sample_per_class]]
            elif task==0:
                train_x = np.concatenate((train_x, x[indx[slot_no*sample_per_class:(slot_no+1)*sample_per_class],:]), axis=0)
                train_y = np.concatenate((train_y, y[indx[slot_no*sample_per_class:(slot_no+1)*sample_per_class]]), axis=0)
            else:
                train_x = np.concatenate((train_x, x[indx[slot_no*sample_per_class:(slot_no+1)*sample_per_class],:]), axis=0)
                tmp = y[indx[slot_no*sample_per_class:(slot_no+1)*sample_per_class]]
                np.random.shuffle(tmp)
                train_y = np.concatenate((train_y, tmp), axis=0)
                
            if class_no==0:
                test_x = x[indx[500:600],:]
                test_y = y[indx[500:600]]
            else:
                test_x = np.concatenate((test_x, x[indx[500:600],:]), axis=0)
                test_y = np.concatenate((test_y, y[indx[500:600]]), axis=0)     
            
    return train_x, train_y, test_x, test_y


# In[ ]:


def experiment(num):
    if num == 0:
        get_ipython().system("./main.py --experiment=splitMNIST --tasks=10 --scenario=task --si --c=0.1 --iters=2000 --name 'tmp_syn1' --tasks_to_complete=10")
    elif num == 1:
        get_ipython().system("./main.py --experiment=splitMNIST --tasks=10 --scenario=task --replay=current --distill --iters=2000 --name 'tmp_syn1' --tasks_to_complete=10")
    elif num == 2:
        get_ipython().system("./main.py --experiment=splitMNIST --tasks=10 --scenario=task --ewc --lambda=5000 --iters=2000 --name 'tmp_syn1' --tasks_to_complete=10")
    elif num == 3:
        get_ipython().system("./main.py --experiment=splitMNIST --tasks=10 --scenario=task --ewc --online --lambda=5000 --gamma=1 --iters=2000 --name 'tmp_syn1' --tasks_to_complete=10")


# In[ ]:


(X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
data_x = np.concatenate([X_train, X_test])
data_y = np.concatenate([y_train, y_test])
data_y = data_y[:, 0]

total_data = len(data_y) 
reshaped_data = np.zeros((total_data,4,32,32),dtype=int)

for i in range(total_data):
    reshaped_data[i,0:3,:,:] = data_x[i,:].reshape(3,32,32)

    
data_x = reshaped_data.reshape(total_data,64,64)


# In[ ]:


alg = ['SI','LwF','EWC','Online_EWC']
filename = './random_label_res'

if not os.path.exists(filename):
    os.mkdir(filename)

num_points_per_task = 500
slot_fold = range(10)
shift_fold = range(1,7,1)
algs = range(4)

for shift in shift_fold:
    for slot in slot_fold:
        train_x, train_y, test_x, test_y = cross_val_data(
            data_x,data_y,num_points_per_task,slot,shift=shift
        )
        train_grp = (torch.tensor(train_x),torch.tensor(np.ravel(train_y)))
        test_grp = (torch.tensor(test_x),torch.tensor(np.ravel(test_y)))
        torch.save(train_grp,'datasets/mnist/MNIST/processed/training.pt')
        torch.save(test_grp,'datasets/mnist/MNIST/processed/test.pt')
            
        for algo in algs:
            print("doing algo {} shift {} slot {}".format(alg[algo],shift,slot))
            experiment(algo)
            err = np.zeros((10,10),dtype=float)

            for jj in range(10):
                err[jj,:] = unpickle('./tmp_syn1/task'+str(jj+1)+'.pickle')

            with open(filename+'/'+alg[algo]+str(shift)+'_'+str(slot)+'.pickle','wb') as f:
                pickle.dump(err,f)
            


# In[ ]:




