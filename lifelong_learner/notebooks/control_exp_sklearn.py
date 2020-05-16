#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import pickle
from itertools import product

from sklearn.ensemble.forest import _generate_unsampled_indices
from sklearn.ensemble.forest import _generate_sample_indices

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from tqdm import tqdm_notebook as tqdm

from joblib import Parallel, delayed

#Infrastructure
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import NotFittedError

#Data Handling
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    NotFittedError,
)
from sklearn.utils.multiclass import check_classification_targets

#Utils
import numpy as np

from tqdm import tqdm
from sklearn.base import clone 

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


#%%
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def homogenize_labels(a):
    u = np.unique(a)
    return np.array([np.where(u == i)[0][0] for i in a])

#%%
def cross_val_data(data_x, data_y, class_idx, task, cv=1):
    x = data_x.copy()
    y = data_y.copy()
    idx = class_idx.copy()
    
    total_cls = range(task*10,(task+1)*10,1)
    
    for i in total_cls:
        indx = np.roll(idx[i],(cv-1)*100)
        
        if i==total_cls[0]:
            train_x = x[indx[0:500],:]
            test_x = x[indx[500:600],:]
            train_y = y[indx[0:500]]
            test_y = y[indx[500:600]]
        else:
            train_x = np.concatenate((train_x, x[indx[0:500],:]), axis=0)
            test_x = np.concatenate((test_x, x[indx[500:600],:]), axis=0)
            train_y = np.concatenate((train_y, y[indx[0:500]]), axis=0)
            test_y = np.concatenate((test_y, y[indx[500:600]]), axis=0)
        
        
    return train_x, train_y, test_x, test_y
# In[90]:


def LF_experiment(train_x, train_y, test_x, test_y, task, ntrees, cv, acorn=None):
    if acorn is not None:
        np.random.seed(acorn)
       
    m = 1000
    RF = BaggingClassifier(
            DecisionTreeClassifier(
                max_depth=30,
                min_samples_leaf=1,
                max_features="auto"
            ),
            n_estimators=ntrees,
            max_samples=0.63,
            n_jobs = -1
        )
    RF.fit(
                        train_x, 
                        homogenize_labels(train_y)
                        )
        
     
    rf_single_task=RF.predict(test_x)
    err = 1 - np.sum(rf_single_task == homogenize_labels(test_y))/m
    print(err)

    return err


# In[91]:


def run_parallel_exp(data_x, data_y, class_idx, ntrees, task):
    
    err = 0
    for i in range(1,7):
        train_x, train_y, test_x, test_y = cross_val_data(data_x, data_y, class_idx, task, cv=i)
        err += LF_experiment(train_x, train_y, test_x, test_y, task, ntrees, i, acorn=task)
        
    err /= 6
    
    with open('../result/taskRF_'+str(task)+'_'+str(ntrees),'wb') as f:
        pickle.dump(err,f)


# In[92]:


n_tasks = 10
train_file = '/data/Jayanta/continual-learning/train'
unpickled_train = unpickle(train_file)
train_keys = list(unpickled_train.keys())
fine_labels = np.array(unpickled_train[train_keys[2]])
labels = fine_labels


# In[93]:


test_file = '/data/Jayanta/continual-learning/test'
unpickled_test = unpickle(test_file)
test_keys = list(unpickled_test.keys())
fine_labels = np.array(unpickled_test[test_keys[2]])
labels_ = fine_labels


# In[94]:


data_x = np.concatenate((unpickled_train[b'data'], unpickled_test[b'data']), axis=0)
data_y = np.concatenate((labels, labels_), axis=0)

class_idx = [np.where(data_y == u)[0] for u in np.unique(data_y)]


# In[ ]:


trees = range(10,300,10)
task = range(0,10)
iterable = product(task,trees)

for task_,ntrees in iterable:
    run_parallel_exp(data_x, data_y, class_idx, ntrees, task_)


# In[77]:


run_parallel_exp(data_x, data_y, class_idx, 100, 1) 

# In[ ]:




