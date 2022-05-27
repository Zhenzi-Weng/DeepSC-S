"""
Created on Tue Jun 23 23:41:39 2020

@author: Zhenzi Weng
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import tensorflow as tf
import numpy as np
import scipy.io as sio
from models import sem_enc_model, chan_enc_model, Chan_Model, chan_dec_model, sem_dec_model

num_cpus = os.cpu_count()
print("Number of CPU cores is", num_cpus)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True) # assign GPU memory dynamically

###############    define global parameters    ###############
def parse_args():
    parser = argparse.ArgumentParser(description="semantic communication systems for speech transmission")
    
    # parameter of frame
    parser.add_argument("--sr", type=int, default=8000, help="sample rate for wav file")
    parser.add_argument("--num_frame", type=int, default=128, help="number of frames in each batch")
    parser.add_argument("--frame_size", type=float, default=0.016, help="time duration of each frame")
    parser.add_argument("--stride_size", type=float, default=0.016, help="time duration of frame stride")
    
    # parameter of semantic coding and channel coding
    parser.add_argument("--sem_enc_outdims", type=list, default=[32, 128, 128, 128, 128, 128, 128],
                        help="output dimension of SE-ResNet in semantic encoder.")
    parser.add_argument("--chan_enc_filters", type=list, default=[128],
                        help="filters of CNN in channel encoder.")
    parser.add_argument("--chan_dec_filters", type=list, default=[128],
                        help="filters of CNN in channel decoder.")
    parser.add_argument("--sem_dec_outdims", type=list, default=[128, 128, 128, 128, 128, 128, 32],
                        help="output dimension of SE-ResNet in semantic decoder.")
    
    # path of tfrecords files
    parser.add_argument("--trainset_tfrecords_path", type=str, default="path of your trainset.tfrecords",
                        help="tfrecords path of trainset.")
    parser.add_argument("--validset_tfrecords_path", type=str, default="path of your validset.tfrecords",
                        help="tfrecords path of validset.")
    
    # parameter of wireless channel
    parser.add_argument("--snr_train_dB", type=int, default=8, help="snr in dB for training.")
    
    # epoch and learning rate
    parser.add_argument("--num_epochs", type=int, default=1000, help="training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size.")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate.")
    
    args = parser.parse_args()
    
    return args

args = parse_args()
print("Called with args:", args)

frame_length = int(args.sr*args.frame_size)
stride_length = int(args.sr*args.stride_size)

if __name__ == "__main__":
    
    ###############    define system model    ###############
    # define semantic encoder
    sem_enc = sem_enc_model(frame_length, stride_length, args)
    print(sem_enc.summary(line_length=160))
    
    # define channel encoder
    chan_enc = chan_enc_model(frame_length, args)
    print(chan_enc.summary(line_length=160))
    
    # define channel model
    chan_layer = Chan_Model(name="Channel_Model")
    
    # define channel decoder
    chan_dec = chan_dec_model(frame_length, args)
    print(chan_dec.summary(line_length=160))
    
    # define semantic decoder
    sem_dec = sem_dec_model(frame_length, stride_length, args)
    print(sem_dec.summary(line_length=160))
    
    # all trainable weights
    weights_all = sem_enc.trainable_weights + chan_enc.trainable_weights +\
                  chan_dec.trainable_weights + sem_dec.trainable_weights
    
    # define MSE loss function
    mse_loss = tf.keras.losses.MeanSquaredError(name="mse_loss")
    
    # define optimizer
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=args.lr)
    
    ###############    define train step and valid step    ###############
    @tf.function
    def train_step(_input, std):
        
        std = tf.cast(std, dtype=tf.float32)
        with tf.GradientTape() as tape:
            _output, batch_mean, batch_var = sem_enc(_input)
            _output = chan_enc(_output)
            _output = chan_layer(_output, std)
            _output = chan_dec(_output)
            _output = sem_dec([_output, batch_mean, batch_var])
            loss_value = mse_loss(_input, _output)
        
        grads = tape.gradient(loss_value, weights_all)  # compute gradients
        optimizer.apply_gradients(zip(grads, weights_all))  # update parameters

        return loss_value
    
    @tf.function
    def valid_step(_input, std):
        
        std = tf.cast(std, dtype=tf.float32)
        _output, batch_mean, batch_var = sem_enc(_input)
        _output = chan_enc(_output)
        _output = chan_layer(_output, std)
        _output = chan_dec(_output)
        _output = sem_dec([_output, batch_mean, batch_var])
        loss_value = mse_loss(_input, _output)

        return loss_value

    ###############    map function to read tfrecords    ###############
    @tf.function
    def map_function(example):

        feature_map = {"wav_raw": tf.io.FixedLenFeature([], tf.string)}
        parsed_example = tf.io.parse_single_example(example, features=feature_map)
    
        wav_slice = tf.io.decode_raw(parsed_example["wav_raw"], out_type=tf.int16)
        wav_slice = tf.cast(wav_slice, tf.float32) / 2**15

        return wav_slice
    
    ###################    create folder to save data    ###################
    common_dir = "path to save the trained data"
    saved_model = common_dir + "saved_model/"
    
    # create files to save train loss
    train_loss_dir = common_dir + "train/"   
    os.makedirs(train_loss_dir)
    train_loss_file = train_loss_dir + "train_loss.mat"
    train_loss_all = []
    
    # create files to save eval loss
    valid_loss_dir = common_dir + "valid/"   
    os.makedirs(valid_loss_dir)
    valid_loss_file = valid_loss_dir + "valid_loss.mat"
    valid_loss_all = []
    
    print("*****************   start train   *****************")
    snr = pow(10, (args.snr_train_dB / 10))
    std = np.sqrt(1 / (2*snr))
    for epoch in range(args.num_epochs):
        ##########################    train    ##########################
        # read .tfrecords file
        trainset = tf.data.TFRecordDataset(args.trainset_tfrecords_path)
        trainset = trainset.map(map_func=map_function, num_parallel_calls=num_cpus) # num_parallel_calls should be number of cpu cores
        trainset = trainset.shuffle(buffer_size=args.batch_size*657, reshuffle_each_iteration=True)
        trainset = trainset.batch(batch_size=args.batch_size)
        trainset = trainset.prefetch(buffer_size=args.batch_size)
        
        # train_loss for each epoch
        train_loss_epoch = []
        train_loss = 0.0
        
        # record the train time for each epoch
        start = time.time()
        
        for step, _input in enumerate(trainset):
            # train step
            loss_value = train_step(_input, std)
            loss_float = float(loss_value)
            train_loss_epoch.append(loss_float)
            
            # Calculate the accumulated train loss value
            train_loss += loss_float
            
        # average train loss for each epoch
        train_loss /= (step + 1)
        # append one epoch loss value
        train_loss_all.append(np.array(train_loss_epoch, dtype=np.float32))
        
        # print log
        log = "train epoch {}/{}, train_loss = {:.06f}, time = {:.06f}"
        print(log.format(epoch + 1, args.num_epochs, train_loss, time.time() - start))
        
        ##########################    valid    ##########################
        # read .tfrecords file
        validset = tf.data.TFRecordDataset(args.validset_tfrecords_path)
        validset = validset.map(map_func=map_function, num_parallel_calls=num_cpus)
        validset = validset.batch(batch_size=args.batch_size)
        validset = validset.prefetch(buffer_size=args.batch_size)
        
        # valid_loss for each epoch
        valid_loss_epoch = []
        valid_loss = 0.0
            
        # record the valid time for each epoch
        start = time.time()
        
        for step, _input in enumerate(validset):
            # valid step
            loss_value = valid_step(_input, std)
            loss_float = float(loss_value)
            valid_loss_epoch.append(loss_float)
            
            # Calculate the accumulated valid loss value
            valid_loss += loss_float
        
        # average valid loss for each epoch
        valid_loss /= (step + 1)
        # append one epoch loss value
        valid_loss_all.append(np.array(valid_loss_epoch, dtype=np.float32))
        
        # print log
        log = "valid epoch {}/{}, valid_loss = {:.06f}, time = {:.06f}"
        print(log.format(epoch + 1, args.num_epochs, valid_loss, time.time() - start))
        print()
        
        ###################    save the train network    ###################
        if (epoch + 1) % 1000 == 0:
            saved_model_dir = os.path.join(saved_model, "{}_epochs".format(epoch + 1))
            os.makedirs(saved_model_dir)
            
            # semantic_encoder
            sem_enc_h5 = saved_model_dir + "/sem_enc.h5"
            sem_enc.save(sem_enc_h5)
            
            # channel_encoder
            chan_enc_h5 = saved_model_dir + "/chan_enc.h5"
            chan_enc.save(chan_enc_h5)
            
            # channel_decoder
            chan_dec_h5 = saved_model_dir + "/chan_dec.h5"
            chan_dec.save(chan_dec_h5)
            
            # semantic_decoder
            sem_dec_h5 = saved_model_dir + "/sem_dec.h5"
            sem_dec.save(sem_dec_h5)
            
            ################    save train loss and valid loss    ################
            if os.path.exists(train_loss_file):
                os.remove(train_loss_file)
            save_train_loss = {}
            train_loss_save = np.array(train_loss_all, dtype=np.float32)
            save_train_loss["train_loss"] = train_loss_save
            sio.savemat(train_loss_file, save_train_loss)
            
            if os.path.exists(valid_loss_file):
                os.remove(valid_loss_file)
            save_valid_loss = {}
            valid_loss_save = np.array(valid_loss_all, dtype=np.float32)
            save_valid_loss["valid_loss"] = valid_loss_save
            sio.savemat(valid_loss_file, save_valid_loss)
