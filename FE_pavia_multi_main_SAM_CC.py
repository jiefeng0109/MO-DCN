"""
"""
import time
import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import cohen_kappa_score
from processing_library import load_data,disorder,next_batch,save_result
from processing_library import contrary_one_hot
from cv2 import *
from PIL import Image
import scipy.io as sio
import tensorflow.contrib.slim as slim
import math
from processing_library import contrary_one_hot,conv_layer_same,save_result, next_batch_pansharp
import tflearn
from metrics import sam, psnr, scc, ergas,mse

# os.environ["CUDA_VISIBLE_DEVICES"]='0'
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

batch_size = 4
step = 1
display_step = 100


index = batch_size

global_step=tf.Variable(step)
learn_rate=tf.train.exponential_decay(0.0005, global_step,100, 0.96, staircase=False)#学习率
learn_rate2= tf.train.exponential_decay(0.00001, global_step,100, 0.96, staircase=False)#学习率
learn_rate2 = 0.00001
num_epoch = 1500
##############################################################################
data_orig_norm, LRHS, PAN, LRHS_train_patch, PAN_train_patch, LRHS_train_gt, LRHS_test_patch, LRHS_test_gt, PAN_test_patch = load_data('Pavia')
pre = sio.loadmat('E:\code\Pansharpening\code\Run and test\deconv_upsampling_norm\\test_results.mat')
Deconv_LRHS_patch = pre['Deconv_train_results']
Deconv_test_patch = pre['Deconv_test_results']
input_dim, H, W = data_orig_norm.shape[2], PAN_train_patch.shape[1], PAN_train_patch.shape[1]
print('train_sample:', LRHS_train_gt.shape[0])

##################################################################################划分、打乱数据集
ratio = int(LRHS_train_gt.shape[1]/LRHS_train_patch.shape[1])
X_train_LRHS = Deconv_LRHS_patch
X_train_PAN = PAN_train_patch
Y_train = LRHS_train_gt

X_test_LRHS = Deconv_test_patch
X_test_PAN = PAN_test_patch
Y_test = LRHS_test_gt
dim_LRHS = X_train_LRHS.shape[3]
dim_PAN = 1

X_train_PAN_HP = np.expand_dims(X_train_PAN,axis = -1)
X_train_LRHS_HP = Deconv_LRHS_patch
X_test_PAN_HP = np.expand_dims(PAN_test_patch,axis = -1)
X_test_LRHS_HP = Deconv_test_patch

X_train_PAN = np.expand_dims(X_train_PAN,axis= -1)
X_test_PAN = np.expand_dims(X_test_PAN, axis= -1)
# ###############################################################################
#
# # fuse_conv = slim.conv2d(fuse, num_outputs=dim_LRHS, kernel_size =[3, 3], rate=1)
# # fuse_image = slim.conv2d(fuse_conv, num_outputs=dim_LRHS, kernel_size =[3, 3], rate=1)
# ################################loss##############################################################
init = tf.global_variables_initializer()
# ###############################################################################
###############################################################################
def SAM_loss(x,y):
    A= tf.reduce_sum(x*y,axis=3)
    B= tf.reduce_sum(x*x,axis=3)
    C= tf.reduce_sum(y*y,axis=3)
    dafen = tf.acos(tf.div(A,tf.sqrt(tf.multiply(B,C))))*180/3.1415926
    dafen = tf.reduce_mean(tf.reduce_mean(dafen,axis=0))
    return dafen

def scc_loss(img1, img2):
    img1 = tf.convert_to_tensor(img1,dtype='float32')
    img2 = tf.convert_to_tensor(img2,dtype='float32')
    imag_flat1 = tf.reshape(img1,[-1,img1.shape[1]*img1.shape[1],img1.shape[-1]])
    imag_flat2 = tf.reshape(img2,[-1, img2.shape[1]*img2.shape[1], img2.shape[-1]])
    m1,m2 = tf.math.reduce_mean(imag_flat1,axis=1,keep_dims=True),tf.math.reduce_mean(imag_flat2,axis=1,keep_dims=True)
    xm, ym = imag_flat1 - m1, imag_flat2 - m2
    r_num = tf.reduce_mean(tf.multiply(xm,ym),axis=1)
    r_den = tf.math.reduce_std(xm, axis=1) * tf.math.reduce_std(ym, axis=1)
    temp = tf.reduce_mean(tf.divide(r_num,r_den),axis=-1)
    temp = tf.reduce_mean(input_tensor=temp,axis=0)
    return temp

def get_all_mse(X_test):
    num = np.shape(X_test)[0]
    index_all = 0
    step_ = batch_size
    y_pred = []
    flag = 0
    while index_all<num:
        if index_all + step_ > num:
            x1, x2, x3, x4 = X_test_LRHS[index_all:, :, :, :], X_test_PAN[index_all:, :, :], X_test_LRHS_HP[index_all:, :, :, :], X_test_PAN_HP[index_all:, :, :]
        else:
            x1, x2, x3, x4 = X_test_LRHS[index_all:(index_all+step_), :, :, :], X_test_PAN[index_all:(index_all+step_), :, :], X_test_LRHS_HP[index_all:(index_all+step_),
                                                                                             :, :, :], X_test_PAN_HP[
                                                                                                       index_all:(index_all+step_), :, :]
        index_all += step_
        temp1 = fuse_image.eval(feed_dict={x_LRHS: x1, x_PAN: x2, x_LRHS_HP: x3, x_PAN_HP: x4})
        if flag == 0:
            y_pred = temp1
            flag = flag + 1
        else:
            y_pred = np.vstack((y_pred,temp1))

    return y_pred
###############################################################################
def get_index(y_pr, Y_test):
    c_sam = 0
    c_cc = 0
    c_psnr = 0
    c_ergas = 0
    c_mse = 0
    for i in range(y_pr.shape[0]):
        # print(scc(y_pr[i], Y_test[i]))
        c_sam = c_sam + sam(y_pr[i], Y_test[i])
        c_cc = c_cc + scc(y_pr[i], Y_test[i])
        c_psnr = c_psnr + psnr(y_pr[i], Y_test[i])
        c_ergas = c_ergas + ergas(y_pr[i], Y_test[i])
        c_mse = c_mse + mse(y_pr[i] , Y_test[i])
    print('CC %f' % (c_cc/y_pr.shape[0]))
    print('SAM %f' % (c_sam/y_pr.shape[0]))
    print('MSE %f' % ( c_mse/y_pr.shape[0]))
    print('ERGAS %f' % (c_ergas/y_pr.shape[0]))
    print('PSNR %f' % (c_psnr/y_pr.shape[0]))
    print('===============================================')
    return c_cc/X_test_LRHS.shape[0], c_sam/X_test_LRHS.shape[0], c_ergas/X_test_LRHS.shape[0], c_psnr/X_test_LRHS.shape[0]



# fuse_image = tf.get_collection('fuse_image')[0]

model_path=os.getcwd()
with tf.Session() as sess:
    print('====================get epsilon======================')
    sess.run(init)
    saver = tf.train.import_meta_graph('model_first_stage.meta', clear_devices=True)
    saver.restore(sess,tf.train.latest_checkpoint(model_path))
    print('model restored!')
    graph = tf.get_default_graph()
    x_LRHS = graph.get_operation_by_name('x_input1').outputs[0]
    x_PAN = graph.get_operation_by_name('x_input2').outputs[0]
    x_LRHS_HP = graph.get_operation_by_name('x_input3').outputs[0]
    x_PAN_HP = graph.get_operation_by_name('x_input4').outputs[0]
    y = graph.get_operation_by_name('y_output').outputs[0]
    fuse_image = tf.get_collection('fuse_image')[0]
    y_pr = get_all_mse(X_test_LRHS)
    _,epsilon,_,_ = get_index(y_pr, Y_test)
    print('====================get epsilon end======================')
    ###############################################################################
zero = tf.constant([0], dtype=tf.float32)
fine_grind_loss = graph.get_operation_by_name('fine_grind_loss').outputs[0]
content_loss2 = graph.get_operation_by_name('content_loss2').outputs[0]
main_loss = graph.get_operation_by_name('main_loss').outputs[0]
epsilon_loss = 1/(scc_loss(fuse_image, y))+ 0.1*(tf.minimum(zero, SAM_loss(fuse_image,y) - epsilon)) + fine_grind_loss + content_loss2
final_op = tf.train.AdamOptimizer(learn_rate2,name='new_').minimize(epsilon_loss)
with tf.Session() as sess:
    print('====================epsilon constraint start======================')
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.import_meta_graph('model_first_stage.meta', clear_devices=True)
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    print('model restored!')
    epoch = 0
    time_train_start = time.clock()
    while epoch<num_epoch:
        batch_LRHS, batch_PAN, batch_LRHS_HP, batch_PAN_HP, batch_y = next_batch_pansharp(X_train_LRHS, X_train_PAN, X_train_LRHS_HP, X_train_PAN_HP, Y_train, index, batch_size)
        sess.run([final_op], feed_dict={x_LRHS: batch_LRHS,x_PAN: batch_PAN, x_LRHS_HP: batch_LRHS_HP, x_PAN_HP: batch_PAN_HP, y: batch_y})
        if step%display_step == 0:
            y_pr = get_all_mse(X_test_LRHS)
            _, _, _, _ = get_index(y_pr, Y_test)
        index = index+batch_size
        step += 1
        if index> (X_train_LRHS.shape[0]):
            index = batch_size
            epoch=epoch+1
            index_train = np.arange(X_train_LRHS.shape[0])
            np.random.shuffle(index_train)
            X_train_LRHS, X_train_PAN, X_train_LRHS_HP, X_train_PAN_HP, Y_train = X_train_LRHS[index_train, :, :, :], X_train_PAN[index_train, :, :] , X_train_LRHS_HP[index_train, :, :, :], X_train_PAN_HP[index_train, :, :], Y_train[index_train, :, :, :]
    time_train_end = time.clock()
    print("Optimization Finished!")
    y_pr = get_all_mse(X_test_LRHS)
    _, _, _, _ = get_index(y_pr, Y_test)
    print('Train time:', time_train_end - time_train_start)
    print('====================epsilon constraint end======================')