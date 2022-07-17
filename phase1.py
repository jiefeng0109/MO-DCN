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

os.environ["CUDA_VISIBLE_DEVICES"]='0'
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

batch_size = 4
step = 1
display_step = 50


index = batch_size

global_step=tf.Variable(step)
learn_rate=tf.train.exponential_decay(0.0001, global_step,100, 0.96, staircase=False)#学习率
learn_rate2 = 0.00005
# learn_rate= 0.0005
# learn_rate = 0.001
num_epoch1 =400
num_epoch2 = 500
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
    imag_flat1 = tf.reshape(img1,[img1.shape[0],img1.shape[1]*img1.shape[1],img1.shape[-1]])
    imag_flat2 = tf.reshape(img2,[img2.shape[0], img2.shape[1]*img2.shape[1], img2.shape[-1]])
    m1,m2 = tf.math.reduce_mean(imag_flat1,axis=1,keep_dims=True),tf.math.reduce_mean(imag_flat2,axis=1,keep_dims=True)
    xm, ym = imag_flat1 - m1, imag_flat2 - m2
    r_num = tf.reduce_mean(tf.multiply(xm,ym),axis=1)
    r_den = tf.math.reduce_std(xm, axis=1) * tf.math.reduce_std(ym, axis=1)
    temp = tf.reduce_mean(tf.divide(r_num,r_den),axis=-1)
    temp = tf.reduce_mean(input_tensor=temp,axis=0)
    return temp

def high_pass_filtering(image, radius, n):
    """
    高通滤波函数
    :param image: 输入图像
    :param radius: 半径
    :param n: ButterWorth滤波器阶数
    :return: 滤波结果
    """
    # 对图像进行傅里叶变换，fft是一个三维数组，fft[:, :, 0]为实数部分，fft[:, :, 1]为虚数部分
    fft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    # 对fft进行中心化，生成的dshift仍然是一个三维数组
    dshift = np.fft.fftshift(fft)

    # 得到中心像素
    rows, cols = image.shape[:2]
    mid_row, mid_col = int(rows / 2), int(cols / 2)

    # 构建ButterWorth高通滤波掩模

    mask = np.zeros((rows, cols, 2), np.float32)
    for i in range(0, rows):
        for j in range(0, cols):
            # 计算(i, j)到中心点的距离
            d = np.sqrt((i - mid_row)*(i - mid_row) + (j - mid_col)* (j - mid_col))
            try:
                mask[i, j, 0] = mask[i, j, 1] = 1 / (1 + (radius / d)* (radius / d))
            except ZeroDivisionError:
                mask[i, j, 0] = mask[i, j, 1] = 0
    # 给傅里叶变换结果乘掩模
    fft_filtering = dshift * mask
    # 傅里叶逆变换
    ishift = np.fft.ifftshift(fft_filtering)
    image_filtering = cv2.idft(ishift)
    image_filtering = cv2.magnitude(image_filtering[:, :, 0], image_filtering[:, :, 1])
    # 对逆变换结果进行归一化（一般对图像处理的最后一步都要进行归一化，特殊情况除外）
    cv2.normalize(image_filtering, image_filtering, 0, 1, cv2.NORM_MINMAX)
    return image_filtering
def combined_static_and_dynamic_shape(tensor):
  """Returns a list containing static and dynamic values for the dimensions.

  Returns a list of static and dynamic values for shape dimensions. This is
  useful to preserve static shapes when available in reshape operation.

  Args:
    tensor: A tensor of any type.

  Returns:
    A list of size tensor.shape.ndims containing integers or a scalar tensor.
  """
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape
def get_channel_attention_module(feature_map, index, inner_units_ratio=0.5):
    """
    CBAM: convolution block attention module, which is described in "CBAM: Convolutional Block Attention Module"
    Architecture : "https://arxiv.org/pdf/1807.06521.pdf"
    If you want to use this module, just plug this module into your network
    :param feature_map : input feature map
    :param index : the index of convolution block attention module
    :param inner_units_ratio: output units number of fully connected layer: inner_units_ratio*feature_map_channel
    :return:feature map with channel and spatial attention
    """
    with tf.variable_scope("cbam_%s" % (index)):
        feature_map_shape = combined_static_and_dynamic_shape(feature_map)
        # channel attention
        channel_avg_weights = tf.nn.avg_pool(
            value=feature_map,
            ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],
            strides=[1, 1, 1, 1],
            padding='VALID'
        )
        channel_max_weights = tf.nn.max_pool(
            value=feature_map,
            ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],
            strides=[1, 1, 1, 1],
            padding='VALID'
        )
        channel_avg_reshape = tf.reshape(channel_avg_weights,
                                         [feature_map_shape[0], 1, feature_map_shape[3]])
        channel_max_reshape = tf.reshape(channel_max_weights,
                                         [feature_map_shape[0], 1, feature_map_shape[3]])
        channel_w_reshape = tf.concat([channel_avg_reshape, channel_max_reshape], axis=1)

        fc_1 = tf.layers.dense(
            inputs=channel_w_reshape,
            units=feature_map_shape[3] * inner_units_ratio,
            name="fc_1",
            activation=tf.nn.relu
        )
        fc_2 = tf.layers.dense(
            inputs=fc_1,
            units=feature_map_shape[3],
            name="fc_2",
            activation=None
        )
        channel_attention = tf.reduce_sum(fc_2, axis=1, name="channel_attention_sum")
        channel_attention = tf.nn.sigmoid(channel_attention, name="channel_attention_sum_sigmoid")
        channel_attention = tf.reshape(channel_attention, shape=[feature_map_shape[0], 1, 1, feature_map_shape[3]])
        return channel_attention
def get_spatial_attention_module(feature_map, index, inner_units_ratio=0.5):
    """
    CBAM: convolution block attention module, which is described in "CBAM: Convolutional Block Attention Module"
    Architecture : "https://arxiv.org/pdf/1807.06521.pdf"
    If you want to use this module, just plug this module into your network
    :param feature_map : input feature map
    :param index : the index of convolution block attention module
    :param inner_units_ratio: output units number of fully connected layer: inner_units_ratio*feature_map_channel
    :return:feature map with channel and spatial attention
    """
    with tf.variable_scope("cbam_%s" % (index)):
        feature_map_shape = combined_static_and_dynamic_shape(feature_map)
        # spatial attention
        channel_wise_avg_pooling = tf.reduce_mean(feature_map, axis=3)
        channel_wise_max_pooling = tf.reduce_max(feature_map, axis=3)

        channel_wise_avg_pooling = tf.reshape(channel_wise_avg_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2],
                                                     1])
        channel_wise_max_pooling = tf.reshape(channel_wise_max_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2],
                                                     1])

        channel_wise_pooling = tf.concat([channel_wise_avg_pooling, channel_wise_max_pooling], axis=3)
        spatial_attention = slim.conv2d(
            channel_wise_pooling,
            1,
            [5, 5],
            padding='SAME',
            activation_fn=tf.nn.sigmoid,
            scope="spatial_attention_conv"
        )
        return spatial_attention

def Selective_attention_Conv(mask1, mask2,  M, r, L=32, stride=1, is_training=True):
    input_feature = mask1.get_shape().as_list()[3]
    with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.relu):
        fea_U = tf.add(mask1, mask2)
        fea_con = tf.stack((mask1, mask2),axis = 3)
        fea_con = tf.squeeze(fea_con,-1)
        f_gp1 = slim.conv2d(fea_U, num_outputs=64, kernel_size = [3, 3], rate=1, stride=stride)
        f_gp2 = slim.conv2d(f_gp1, num_outputs=1, kernel_size = [3, 3], rate=1, stride=stride)
        for _ in range(M):
            fcs = slim.conv2d(f_gp2, num_outputs=1, kernel_size = [3, 3], rate=1, stride=stride,activation_fn = None)
            if _ == 0:
                att_vec = fcs
            else:
                att_vec = tf.stack((att_vec, fcs), axis = 3)
        att_vec_softmax = tf.nn.softmax(att_vec, axis = 3)
        att_vec_softmax = tf.squeeze(att_vec_softmax,-1)
        final_feature = tf.multiply(fea_con, att_vec_softmax)
        # fea_v = tf.multiply
        final = tf.reduce_sum(final_feature, axis = -1)
        attention_expand = tf.reshape(tf.concat([final] * dim_LRHS, axis=1), [-1, H, H, dim_LRHS])
    return attention_expand

def content_inference(x_PAN, x_LRHS):
    net1 = slim.conv2d(x_LRHS, num_outputs=64, kernel_size = [1, 1], rate=1)
    net1 = slim.conv2d(net1, num_outputs=64, kernel_size = [1, 1], rate=1)

    image_concat = tf.concat([x_PAN, net1],3)
    spa_spec_infer = slim.conv2d(image_concat, num_outputs=64, kernel_size = [3, 3], rate=1)
    # spa_spec_infer = slim.conv2d(spa_spec_infer, num_outputs=64, kernel_size =[3, 3], rate=1)
    # spa_spec_infer = slim.conv2d(spa_spec_infer, num_outputs=64, kernel_size =[3, 3], rate=1)

    # spa_spec_infer = tf.concat([net1, spa_spec_infer],3)
    spa_spec_infer = tf.add(net1, spa_spec_infer)
    spec_pred = slim.conv2d(spa_spec_infer, num_outputs=64, kernel_size =[1, 1], rate=1)
    # spec_pred = slim.conv2d(spec_pred, num_outputs=dim_LRHS, kernel_size =[1, 1], rate=1)
    return spec_pred
def content_inference1(x_PAN, x_LRHS):
    # net1 = slim.conv2d(x_LRHS, num_outputs=64, kernel_size = [1, 1], rate=1)
    # net1 = slim.conv2d(net1, num_outputs=64, kernel_size = [1, 1], rate=1)

    image_concat = tf.concat([x_PAN, x_LRHS],3)
    spa_spec_infer = slim.conv2d(image_concat, num_outputs=dim_LRHS, kernel_size = [3, 3], rate=1)
    spa_spec_infer = slim.conv2d(spa_spec_infer, num_outputs=dim_LRHS, kernel_size =[3, 3], rate=1)
    # spa_spec_infer = slim.conv2d(spa_spec_infer, num_outputs=dim_LRHS, kernel_size =[3, 3], rate=1)
    # spa_spec_infer = slim.conv2d(spa_spec_infer, num_outputs=dim_LRHS, kernel_size =[3, 3], rate=1)

    return spa_spec_infer

X_train_PAN_HP = np.expand_dims(X_train_PAN,axis = -1)
X_train_LRHS_HP = Deconv_LRHS_patch
X_test_PAN_HP = np.expand_dims(PAN_test_patch,axis = -1)
X_test_LRHS_HP = Deconv_test_patch
#####################################################################################################################
x_LRHS = tf.placeholder(tf.float32,[None, X_train_LRHS.shape[1], X_train_LRHS.shape[2], X_train_LRHS.shape[3]], name='x_input1')
x_PAN = tf.placeholder(tf.float32,[None, X_train_PAN.shape[1], X_train_PAN.shape[2], 1], name='x_input2')
x_LRHS_HP = tf.placeholder(tf.float32,[None, X_train_LRHS.shape[1], X_train_LRHS.shape[2], X_train_LRHS.shape[3]], name='x_input3')
x_PAN_HP = tf.placeholder(tf.float32,[None, X_train_PAN.shape[1], X_train_PAN.shape[2],1], name='x_input4')
y = tf.placeholder(tf.float32,[None, LRHS_train_gt.shape[1],LRHS_train_gt.shape[2],LRHS_train_gt.shape[3]], name='y_output')

###############################细节提取分支
#####PAN分支
conv1_1 = slim.conv2d(x_PAN_HP, num_outputs=dim_LRHS, kernel_size = [3, 3], rate=1)
conv1_2 = slim.conv2d(conv1_1, num_outputs=dim_LRHS, kernel_size = [3, 3], rate=1)
# conv1_3 = slim.conv2d(conv1_2, num_outputs=dim_LRHS, kernel_size = [3, 3], rate=1)

#####HSI分支
conv2_1 = slim.conv2d(x_LRHS_HP, num_outputs=dim_LRHS, kernel_size = [3, 3], rate=1)
conv2_2 = slim.conv2d(conv2_1, num_outputs=dim_LRHS, kernel_size = [3, 3], rate=1)
# conv2_3 = slim.conv2d(conv2_2, num_outputs=dim_LRHS, kernel_size = [3, 3], rate=1)

spatial_attention_PAN = get_spatial_attention_module(conv1_2,1)
spatial_attention_HSI = get_spatial_attention_module(conv2_2,2)
channel_attention = get_channel_attention_module(conv2_2,1)
spatial_attention = Selective_attention_Conv(mask1=spatial_attention_PAN, mask2=spatial_attention_HSI, M=2, r=6, is_training=True )
fea_att = tf.multiply(tf.add(conv1_2,conv2_2), spatial_attention)
fea_att = tf.multiply(fea_att, channel_attention)
fea_att = slim.conv2d(fea_att, num_outputs=dim_LRHS, kernel_size =[3, 3], rate=1)
# fea_att = slim.conv2d(fea_att, num_outputs=dim_LRHS, kernel_size =[3, 3], rate=1)
################################content 分支(HyperPNN2)
content = content_inference1(x_PAN, x_LRHS)
content_y = content_inference1(x_PAN, y)

fuse_image = tf.add(content,fea_att,name = 'fuse_image')
# fuse_conv = slim.conv2d(fuse, num_outputs=dim_LRHS, kernel_size =[3, 3], rate=1)
# fuse_image = slim.conv2d(fuse_conv, num_outputs=dim_LRHS, kernel_size =[3, 3], rate=1)
################################loss##############################################################
content_loss2 = tf.reduce_mean(tf.square(y-content),name='content_loss2')
fine_grind_loss = tf.reduce_mean(tf.square(fea_att-(y-content),name='fine_grind_loss'))
main_loss = tf.reduce_mean(tf.square(tf.subtract(fuse_image,y)),name='main_loss')
all_loss =  tf.add(tf.add(main_loss,fine_grind_loss),content_loss2,name='all_loss')
all_loss2 = -scc_loss(fuse_image,y)+ fine_grind_loss +content_loss2

train_step = tf.train.AdamOptimizer(learn_rate).minimize(all_loss2)
init = tf.global_variables_initializer()
###############################################################################
X_train_PAN = np.expand_dims(X_train_PAN,axis= -1)
X_test_PAN = np.expand_dims(X_test_PAN, axis= -1)

def get_all_mse(X_test):
    num = np.shape(X_test)[0]
    index_all = 0
    step_ = 2
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
    return c_cc/X_test_LRHS.shape[0], c_sam/X_test_LRHS.shape[0], c_ergas/X_test_LRHS.shape[0], c_psnr/X_test_LRHS.shape[0], c_mse/X_test_LRHS.shape[0]


saver = tf.train.Saver()
tf.add_to_collection('fuse_image', fuse_image)
run_time = 1
for num_run in range(run_time):
    with tf.Session() as sess:
        sess.run(init)
        epoch = 0
        time_train_start = time.clock()
        print('run_time:', run_time)
        print('===================pahse1:MSE LOSS==============================')
        while epoch<num_epoch1:
            batch_LRHS, batch_PAN, batch_LRHS_HP, batch_PAN_HP, batch_y = next_batch_pansharp(X_train_LRHS, X_train_PAN, X_train_LRHS_HP, X_train_PAN_HP, Y_train, index, batch_size)
            sess.run([train_step], feed_dict={x_LRHS: batch_LRHS,x_PAN: batch_PAN, x_LRHS_HP: batch_LRHS_HP, x_PAN_HP: batch_PAN_HP, y: batch_y})
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
        RMSE = tf.sqrt(tf.reduce_mean(tf.square(y_pr - Y_test)))
        c_cc, c_sam, c_ergas, c_psnr, epsilon = get_index(y_pr, Y_test)
        print('Train time:',time_train_end - time_train_start)
        saver.save(sess, 'model_first_stage')
        print('Save done')
        print('===================pahse1:end==============================')
    ###############################################################################
