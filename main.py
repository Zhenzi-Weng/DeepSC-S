"""
Created on Tue Jun 23 23:41:39 2020

@author: Zhenzi Weng
"""

import time
import sys
import tensorflow as tf
import librosa
import argparse
import numpy as np
import scipy.io as sio
import os
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.framework import arg_scope
from speech_frame import enframe
from speech_frame import deframe
from file_path import path_dir
from SE_model import SE_ResNeXt
from random import choice


gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True) # assign GPU memory dynamically
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# recoring print content
class Logger(object):
    def __init__(self, fileN="record.txt"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()                 
    def flush(self):
        self.log.flush()
 
sys.stdout = Logger("./training_record.txt")


""" define global parameters """
def parse_args():
    parser = argparse.ArgumentParser(description="semantic communication systems for speech transmission")
    
    # parameter for mel spectrogram
    parser.add_argument("--sr", type=int, default=8000, help="sample rate for wav file")
    parser.add_argument("--num_frames", type=int, default=128, help="number of frame in each barch")
    parser.add_argument("--frame_length", type=float, default=0.016, help="the time duration of frame")
    parser.add_argument("--frame_shift", type=float, default=0.016, help="the time duration of frame shift") 

    # channel number and bit number
    parser.add_argument("--bit_num", type=int, default=512, help="number of bits")
    parser.add_argument("--channel_num", type=int, default=512, help="number of channel uses")

    # learning rate
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning_rate for training")
    
    # EbN0 set
    parser.add_argument("--snr_train_db", type=float, default=8, help="snr for training")
    
    args = parser.parse_args()
    
    return args

args = parse_args()
print("Called with args:")
print(args)


path_training_set = "./dataset/clean_trainset/"
print(path_training_set)


# index the wav file for each iteration
fname = os.listdir(path_training_set)
fname = [x for x in fname if x.endswith(".wav")]
num_training_examples = len(fname)

batch_size = 8
num_epochs = 400
num_batches_per_epoch = int( num_training_examples / batch_size )

length = int(args.sr * args.frame_length)
shift = int(args.sr * args.frame_shift)
len_one = args.num_frames * shift + length - shift

bits_per_symbol = 6
rate = 1 / 3
PCM_bits = 8
L_total = 512
chan_filters = int ( 2 * PCM_bits / rate / bits_per_symbol )

assert args.num_frames*length*chan_filters//2 % L_total == 0, "check the size of frames_input"



def W_size():
    size = np.zeros(2, dtype=int)
    size[0] = 5
    size[1] = 5

    return size


def Batch_Normalization(inputs, training, scope):
    # batch normalization
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=inputs, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=inputs, is_training=training, reuse=True))


def forw_conv(inputs, kernel, stride, filters, is_train, layer_name):
    # convolution
    with tf.name_scope(layer_name + "_conv"):
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
        bias_initializer = tf.constant_initializer(value=0.0)
        
        conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel, strides=stride,
                                      kernel_initializer=kernel_initializer, 
                                      bias_initializer = bias_initializer,
                                      padding="SAME")(inputs)
        
    # batch normalization
    conv_BN = Batch_Normalization(conv, training = is_train, scope = layer_name + "_BN")
        
    return conv_BN


def encoder(frames_input, is_train):
    
    # source coding
    source_input = tf.reshape(frames_input, [tf.shape(frames_input)[0], args.num_frames, length, 1])
    with tf.compat.v1.variable_scope("enc_s1"): 
        s1 = SE_ResNeXt(source_input, filters = 32, out_dim = 32, is_train = is_train, name = "enc_s1", blocks = 6).model
      
    # channel coding
    with tf.compat.v1.variable_scope("enc_c1"):               
        c1 = forw_conv(s1, [W_size()[0], W_size()[1]], [1, 1], chan_filters, is_train=is_train, layer_name = "enc_c1")
    
    # reshape layer
    x = tf.reshape(c1, [tf.shape(c1)[0], args.num_frames*length*chan_filters // L_total // 2, L_total, 2], name = "x")
    
    # normalize the average power of each dim in x into 0.5
    x_norm = tf.scalar_mul(tf.sqrt(tf.cast(tf.shape(x)[2] / 2, tf.float32)), tf.nn.l2_normalize(x, axis=2))
    x_norm = tf.identity(x_norm, name = "x_norm")
    
    return x_norm


# channel layer 
def channel_layer(x, std):
    x_real = x[:, :, :, 0]
    x_imag = x[:, :, :, 1]
    x_complex = tf.complex( real=x_real, imag=x_imag )
    
    # channel
    h_real = tf.divide( tf.random.normal(shape=[tf.shape(x_complex)[0], args.num_frames*length*chan_filters // L_total // 2, 1], dtype=tf.float32),
                        tf.sqrt(2.) )
    h_imag = tf.divide( tf.random.normal(shape=[tf.shape(x_complex)[0], args.num_frames*length*chan_filters // L_total // 2, 1], dtype=tf.float32),
                        tf.sqrt(2.) )
    h_real = tf.add( tf.sqrt(3. / 4.), tf.multiply(tf.sqrt(1. / 4.), h_real) )
    h_imag = tf.add( tf.sqrt(3. / 4.), tf.multiply(tf.sqrt(1. / 4.), h_imag) )
    
    h_complex = tf.complex( real=h_real, imag=h_imag )
    
    # noise
    noise = tf.complex(
        real = tf.random.normal(shape=tf.shape(x_complex), mean=0.0, stddev=std, dtype=tf.float32),
        imag = tf.random.normal(shape=tf.shape(x_complex), mean=0.0, stddev=std, dtype=tf.float32))
    
    # received signal y
    hx = tf.multiply( h_complex, x_complex )
    r_complex = tf.add( hx, noise )
     
    # perfect channel estimation
    y_complex = tf.divide( r_complex, h_complex )
    y_real = tf.reshape( tf.math.real(y_complex), [tf.shape(y_complex)[0], args.num_frames*length*chan_filters // L_total // 2, L_total, 1] )
    y_imag = tf.reshape( tf.math.imag(y_complex), [tf.shape(y_complex)[0], args.num_frames*length*chan_filters // L_total // 2, L_total, 1] )
    y = tf.concat( [y_real, y_imag], -1, name = "y" )
    
    return y


def decoder(y, is_train):
    # reshape layer
    channel_input = tf.reshape(y, [tf.shape(y)[0], args.num_frames, length, chan_filters])
    
    # channel decoding
    with tf.compat.v1.variable_scope("dec_c1"):               
        c1 = forw_conv(channel_input, [W_size()[0], W_size()[1]], [1, 1], chan_filters, is_train=is_train, layer_name = "dec_c1")
        c1 = tf.nn.relu(c1)
        
    # source decoding
    with tf.compat.v1.variable_scope("dec_s1"): 
        s1 = SE_ResNeXt(c1, filters = 32, out_dim = 32, is_train = is_train, name = "dec_s1", blocks = 6).model
        
    with tf.compat.v1.variable_scope("dec_s2"):               
        s2 = forw_conv(s1, [W_size()[0], W_size()[1]], [1, 1], 1, is_train=is_train, layer_name = "dec_s2")   
     
    frames_output = tf.reshape(s2, [tf.shape(s2)[0], args.num_frames, length], name = "frames_output") 
    
    return frames_output
    

def train_optim(loss, lr):
    
    with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
        """optimizer"""
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.99, epsilon=1e-8)
        auto_solver = optimizer.minimize(loss)
    
    return auto_solver


def norm(frames): 
    # mean and var
    batch_mean, batch_var = tf.nn.moments(frames, [1, 2])
    mean = tf.tile(tf.reshape(batch_mean, [tf.shape(batch_mean)[0], 1, 1]), [1, args.num_frames, length]) 
    var = tf.tile(tf.reshape(batch_var, [tf.shape(batch_var)[0], 1, 1]), [1, args.num_frames, length])
    
    frames_norm = tf.divide(tf.subtract(frames, mean), tf.sqrt(var), name = "frames_norm")
    
    return frames_norm, mean, var


def denorm(frames, mean, var): 
    
    frames_denorm = tf.add(tf.multiply(frames, tf.sqrt(var)), mean, name = "frames_denorm")
    
    return frames_denorm


def wav_generator(data_path):    
    
    wav_all = np.zeros(shape=[batch_size, len_one])
    
    ind = 0
    
    while ind < batch_size:
        
        current_file_path = path_dir(data_path)
        wav_signal, _ = librosa.load(current_file_path, sr = args.sr)
         
        if len(wav_signal) <= len_one:      
            while len(wav_signal) <= len_one:
                wav_signal = np.concatenate((wav_signal, wav_signal), axis = 0)
            wav_all[ind] = wav_signal[0 : len_one]
            ind += 1
            
        else:
            parts = len(wav_signal) // len_one + 1 
            wav_signal = np.concatenate( (wav_signal, wav_signal), axis = 0 )
            start_ind = choice( np.arange(parts) )
            num_mb = np.min( [parts - start_ind, batch_size - ind] )
            wav_t = wav_signal[start_ind * len_one : (start_ind + num_mb) * len_one]
            wav_all[ ind:ind + num_mb ] = np.reshape( wav_t, [num_mb, len_one] )           
            ind += num_mb
        
    return wav_all


""" start main function"""
print("start main function")
wav_input = tf.compat.v1.placeholder(tf.float32, shape=[None, len_one], name="wav_input")
noise_std = tf.compat.v1.placeholder(tf.float32, shape=[], name = "std") 
lr = tf.compat.v1.placeholder(tf.float32, shape=[], name = "learning_rate") 
is_train = tf.compat.v1.placeholder_with_default(False, shape =[], name = "is_train")


# frame_input
frames_input = enframe(wav_input, length, shift, args.num_frames)

# frames norm
frames_norm, mean, var = norm(frames_input)

# autoencoder
x = encoder(frames_norm, is_train)
y = channel_layer(x, noise_std)
frames_output = decoder(y, is_train)

# frames_denorm
frames_denorm = denorm(frames_output, mean, var)
wav_output = deframe(frames_denorm, length, shift, args.num_frames)

# loss computation
loss = tf.compat.v1.losses.mean_squared_error(wav_input, wav_output, scope = "loss")

# 
print(wav_input)
print(frames_input)
print(frames_norm)
print(x)
print(y)
print(frames_output)
print(frames_denorm)
print(wav_output)
print(loss)


# Optimizer
auto_solver = train_optim(loss, lr)


# initialization
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
saver = tf.compat.v1.train.Saver()

# parameters
R = args.bit_num/args.channel_num
snr_train = pow(10, (args.snr_train_db/10))
std_train = np.sqrt(1 / (2 * R * snr_train))

# training stage
with tf.compat.v1.Session(config=config) as sess:
    # initialization of NN parameters
    sess.run(tf.compat.v1.global_variables_initializer())
    
    # create a files to save the network
    common_dir = "./" + "saved_data/"
    os.makedirs(common_dir + "saved_network/") 
    trained_network = common_dir + "saved_network/" + "trained_net.ckpt"

    # create a mat files to save the training loss 
    os.makedirs(common_dir + "/loss")   
    training_loss_all = common_dir + "/loss/" + "training_loss" + ".mat"    
    save_training_loss_all = {}
    training_loss = np.zeros([num_batches_per_epoch, num_epochs], dtype = float)
    
    # start training 
    epoch_lr = args.learning_rate
    
    for epo in range(num_epochs):
        train_loss = 0
        start = time.time()
        
        # index the wav file for each epochs
        for batch in range(num_batches_per_epoch):
            
            data_path = path_training_set  
            wav_all = wav_generator(data_path)               
            batch_loss, _ = sess.run([loss, auto_solver], feed_dict={wav_input: wav_all, 
                                                                     noise_std: std_train,
                                                                     lr: epoch_lr,
                                                                     is_train: True})
             
            training_loss[batch, epo] = batch_loss
            train_loss += batch_loss * batch_size                 
        
        train_loss /= batch_size * num_batches_per_epoch
        
        print("epochs:", epo + 1, "train loss:", train_loss, "time:", time.time() - start)
            
             
    # save the network
    saver_path = saver.save(sess, trained_network)  
    print("saved meta network:", saver_path)
            
    # save loss_all    
    save_training_loss_all["training_loss"] = training_loss
    sio.savemat(training_loss_all, save_training_loss_all)
    
    