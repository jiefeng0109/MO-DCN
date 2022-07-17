# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 20:38:39 2018

@author: Jiantong Chen
"""

import os
import copy

import scipy.io as sio
import tensorflow as tf
import numpy as np

###############################################################################
def load_data(data_name):
    '''读取数据'''
    path = os.getcwd()
    pre = sio.loadmat(path + '/' + data_name + '_pre.mat')
    
    data_orig_norm = pre['Orig_norm']
    LRHS = pre['LRHS']
    PAN = pre['PAN']

    LRHS_train_patch = pre['LRHS_train_patch']
    PAN_train_patch = pre['PAN_train_patch']
    LRHS_train_gt = pre['LRHS_train_gt']

    LRHS_test_patch = pre['LRHS_test_patch']
    LRHS_test_gt = pre['LRHS_test_gt']
    PAN_test_patch = pre['PAN_test_patch']


    # LRHS_loc = pre['LRHS_loc']
    # PAN_loc = pre['PAN_loc']

    return data_orig_norm,LRHS,PAN,LRHS_train_patch,PAN_train_patch,LRHS_train_gt,LRHS_test_patch,LRHS_test_gt,PAN_test_patch

###############################################################################
def disorder(X,Y,W,A,Z):
    '''打乱顺序'''
    index_train = np.arange(X.shape[0])
    np.random.shuffle(index_train)
    X = X[index_train, :]
    Y = Y[index_train, :]
    W = W[index_train, :]
    A = A[:, index_train]
    Z = Z[:, index_train]

    return X,Y,W,A,Z
###############################################################################
def next_batch_pansharp(image1,image2,image3,image4, lable,batch_size,aaa):
    '''数据分批'''
    start = batch_size-aaa
    end = batch_size
    return image1[start:end], image2[start:end], image3[start:end], image4[start:end],lable[start:end]

def next_batch(image,lable,batch_size,aaa):
    '''数据分批'''
    start = batch_size-aaa
    end = batch_size
    return image[start:end,:,:,:],lable[start:end,:,:,:]

def next_batch_CAVE(image,label,batch_size,aaa,index_train):
    '''数据分批'''
    a1 = index_train[batch_size-aaa]
    a2 = index_train[batch_size-aaa+1]
    image1 = image[a1]
    image2 = image[a2]
    label1 = label[a1]
    label2 = label[a2]

    return np.stack((image1,image2)),np.stack((label1,label2))

def next_batch_three(image1,image2, lable,batch_size):
    '''数据分批'''
    start = batch_size-16
    end = batch_size
    return image1[start:end,:,:,:],image2[start:end,:,:,:],lable[start:end,:,:,:]

###############################################################################
def conv_layer_same(x,W,B,stride):
    '''不改变特征图尺寸的卷积'''
    x = tf.nn.conv2d(x,W,stride,padding='SAME',name='CONV')
    h = tf.nn.bias_add(x,B)
    bn = tf.contrib.layers.batch_norm(h,decay=0.9,epsilon=1e-5,scale=True,is_training=True)
    convout = tf.nn.relu(bn)
    return convout
###############################################################################
def conv_layer_valid(x,W,B,stride):
    '''改变特征图尺寸的卷积'''
    x = tf.nn.conv2d(x,W,stride,padding='VALID',name='CONV')
    h = tf.nn.bias_add(x,B)
    bn = tf.contrib.layers.batch_norm(h,decay=0.9,epsilon=1e-5,scale=True,is_training=True)
    convout = tf.nn.relu(bn)
    return convout
###############################################################################
def contrary_one_hot(label):
    '''将onehot标签转化为真实标签'''
    size=len(label)
    label_ori=np.empty(size)
    for i in range(size):
        label_ori[i]=np.argmax(label[i])+1
    return label_ori
###############################################################################
def save_result(data_name,oa,aa,kappa,per_class_acc,train_time,test_time):
    '''将实验结果保存在txt文件中'''
    write_content='\n'+data_name+'\n'+'oa:'+str(oa)+' aa:'+str(aa)+' kappa:'+str(kappa)+'\n'+'per_class_acc:'+str(per_class_acc)+'\n'+'train_time:'+str(train_time)+' test_time:'+str(test_time)+'\n'
    f = open(os.getcwd()+'/实验结果.txt','a')
    f.writelines(write_content)
    f.close()
    return       