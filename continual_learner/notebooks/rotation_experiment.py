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

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



def dehomogenize(labels, task=1):
    u = np.unique(labels)
    return np.array([(task-1)*10+np.where(u == i)[0][0] for i in labels])



def shape_input(pic):
    l = pic.shape[1]
    pic_ = np.uint8(np.zeros((l,l,3),dtype=int))
    
    pic_[:,:, 0] = pic[0,:,:]
    pic_[:,:, 1] = pic[1,:,:]
    pic_[:,:, 2] = pic[2,:,:]
    
    return pic_

def cross_val_data(data_x, data_y, total_cls=10):
    
    x = data_x.copy()
    y = data_y.copy()
    idx = [np.where(data_y == u)[0] for u in np.unique(data_y)]
    
    
    for i in range(total_cls):
        indx = idx[i] #np.roll(idx[i],(cv-1)*100)
        random.shuffle(indx)
        
        if i==0:
            train_x1 = x[indx[0:250],:]
            train_x2 = x[indx[250:500],:]
            train_y1 = y[indx[0:250]]
            train_y2 = y[indx[250:500]]
            
            test_x1 = x[indx[500:600],:]
            test_y1 = y[indx[500:600]]
        else:
            train_x1 = np.concatenate((train_x1, x[indx[0:250],:]), axis=0)
            train_x2 = np.concatenate((train_x2, x[indx[250:500],:]), axis=0)
            train_y1 = np.concatenate((train_y1, y[indx[0:250]]), axis=0)
            train_y2 = np.concatenate((train_y2, y[indx[250:500]]), axis=0)
            
            test_x1 = np.concatenate((test_x1, x[indx[500:600],:]), axis=0)
            test_y1 = np.concatenate((test_y1, y[indx[500:600]]), axis=0)
        
    
    #adding unnnecessary data for 10 tasks, otherwise it breaks the code
    demo_data = np.random.rand(2500*8,3072)
    demo_test_data = np.random.rand(1000*9,3072)
    
    train_y = np.concatenate((np.ravel(train_y1),dehomogenize(train_y2,task=2)),axis=0)
    test_y = np.concatenate((np.ravel(test_y1),dehomogenize(test_y1,task=2)),axis=0)
    
    for i in range(3,11):
        train_y = np.concatenate((train_y,dehomogenize(train_y2,task=i)),axis=0)
        test_y = np.concatenate((test_y,dehomogenize(test_y1,task=i)),axis=0)
    
    train_x = np.concatenate((train_x1,train_x2,demo_data),axis=0)
    test_x = np.concatenate((test_x1,demo_test_data),axis=0)
    
    return train_x, train_y, test_x, test_y 

def image_aug(pic, angle, centroid_x=23, centroid_y=23, win=16, scale=1.45):
    im_sz = int(np.floor(pic.shape[0]*scale))
    pic_ = np.uint8(np.zeros((im_sz,im_sz,3),dtype=int))
    
    pic_[:,:,0] = ndimage.zoom(pic[:,:,0],scale)
    pic_[:,:,1] = ndimage.zoom(pic[:,:,1],scale)
    pic_[:,:,2] = ndimage.zoom(pic[:,:,2],scale)
    
    image_aug = rotate(pic_, angle, resize=False)
    #print(image_aug.shape)
    image_aug_ = image_aug[centroid_x-win:centroid_x+win,centroid_y-win:centroid_y+win,:]
    
    return img_as_ubyte(image_aug_)


def run_nn(num):
    if num == 0:
        get_ipython().system("../main.py --experiment=splitMNIST --tasks=10 --scenario=task --si --c=0.1 --iters=2000 --name 'tmp' --tasks_to_complete=2")
    elif num == 1:
        get_ipython().system("../main.py --experiment=splitMNIST --tasks=10 --scenario=task --replay=current --distill --iters=2000 --name 'tmp' --tasks_to_complete=2")
    elif num == 2:
        get_ipython().system("../main.py --experiment=splitMNIST --tasks=10 --scenario=task --ewc --lambda=5000 --iters=2000 --name 'tmp' --tasks_to_complete=2")
    elif num == 3:
        get_ipython().system("../main.py --experiment=splitMNIST --tasks=10 --scenario=task --ewc --online --lambda=5000 --gamma=1 --iters=2000 --name 'tmp' --tasks_to_complete=2")

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

angles = np.arange(0,360,1)
cvs = np.arange(1,7)
task_label = range(0,10)

data_x = np.concatenate((unpickled_train[b'data'], unpickled_test[b'data']), axis=0)
data_y = np.concatenate((labels,labels_), axis=0)

class_idx = [np.where(data_y == u)[0] for u in np.unique(data_y)]

data_x1 = np.zeros((6000,3072), dtype=int)
data_y1 = np.zeros((6000,1), dtype=int)
ind = 0

for j in range(10):
    for i in range(600):
        data_x1[ind,:] = data_x[class_idx[j][i]]
        data_y1[ind,:] = data_y[class_idx[j][i]]
        ind += 1


def run_exp(data_x1, data_y1, angle, reps=1, acorn=None):
    if acorn is not None:
        np.random.seed(acorn)
    
    errors = np.zeros((4,2),dtype=float)
    task1_file = '../tmp/task1.pickle'
    task2_file = '../tmp/task2.pickle'
    
    for rep in range(reps):
        train_x, train_y, test_x, test_y = cross_val_data(data_x1, data_y1, total_cls=10)
    
    
        #change data angle for second task
        tmp_data = train_x[2500:2*2500,:].copy()
        _tmp_ = np.zeros((3,32,32), dtype=int)
        total_data = tmp_data.shape[0]
        
        for i in range(total_data):
            tmp_ = image_aug(shape_input(tmp_data[i].reshape(3,32,32)),angle)
        
            _tmp_[0,:,:] = tmp_[:,:,0]
            _tmp_[1,:,:] = tmp_[:,:,1]
            _tmp_[2,:,:] = tmp_[:,:,2]
        
            tmp_data[i,:] = _tmp_.flatten()
            
        train_x[2500:2*2500,:] = tmp_data.copy()
        
        #change datashape to feed to the codes
        total_train_data = train_y.shape[0]
        reshaped_data = np.zeros((total_train_data,4,32,32),dtype=int)
        for i in range(total_train_data):
            tmp = train_x[i,:].reshape(3,32,32)
            reshaped_data[i,0:3,:,:] = tmp
            
        train_x = reshaped_data.reshape(total_train_data,64,64)
        ########################################
        total_test_data = test_y.shape[0]
        reshaped_data = np.zeros((total_test_data,4,32,32),dtype=int)
        for i in range(total_test_data):
            tmp = test_x[i,:].reshape(3,32,32)
            reshaped_data[i,0:3,:,:] = tmp
            
        test_x = reshaped_data.reshape(total_test_data,64,64)
        ########################################
        
        train_data = (torch.tensor(train_x),torch.tensor(np.ravel(train_y)))
        test_data = (torch.tensor(test_x),torch.tensor(np.ravel(test_y)))
        
        torch.save(train_data,'../datasets/mnist/MNIST/processed/training.pt')
        torch.save(test_data,'../datasets/mnist/MNIST/processed/test.pt')
        
        for alg in range(4):
            run_nn(alg)
            err = unpickle(task1_file)
            errors[alg][0] += err[0]
            err = unpickle(task2_file)
            errors[alg][1] += err[0]
            
    errors = errors/reps
    print(errors,angle)
    with open('../rotation_res/'+'SI'+'_'+str(angle)+'.pickle', 'wb') as f:
        pickle.dump(errors[0], f)
        
    with open('../rotation_res/'+'LwF'+'_'+str(angle)+'.pickle', 'wb') as f:
        pickle.dump(errors[1], f)
    
    with open('../rotation_res/'+'EWC'+'_'+str(angle)+'.pickle', 'wb') as f:
        pickle.dump(errors[2], f)
        
    with open('../rotation_res/'+'Online_EWC'+'_'+str(angle)+'.pickle', 'wb') as f:
        pickle.dump(errors[3], f)


angles = np.arange(136,180,1)

for angle in angles:
    run_exp(data_x1, data_y1, angle, reps=20, acorn=1)

get_ipython().system("sudo shutdown now")
