# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 14:35:39 2021

@author: Zhenzi Weng
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
from scipy.io import wavfile
import argparse
import timeit
import sys
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True) # assign GPU memory dynamically

###############    define global parameters    ###############
def parse_args():
    parser = argparse.ArgumentParser(description="Convert the set of .wavs to .TFRecords")
    
    parser.add_argument("--sr", type=int, default=8000, help="sample rate for wav file")
    parser.add_argument("--num_frame", type=int, default=128, help="number of frame in each barch")
    parser.add_argument("--frame_size", type=float, default=0.016, help="time duration of each frame")
    parser.add_argument("--stride_size", type=float, default=0.016, help="time duration of frame stride") 
    
    parser.add_argument("--wav_path", type=str, default="path of your original .wav files",
                        help="path of wavset")
    parser.add_argument("--save_path", type=str, default="path you want to save .tfrecords files",
                        help="path to save .tfrecords file")
    parser.add_argument("--valid_percent", type=float, default=0.05, help="percent of validset in total dataset")
    parser.add_argument("--trainset_filename", type=str, default="trainset.tfrecords", help=".tfrecords filename of trainset")
    parser.add_argument("--validset_filename", type=str, default="validset.tfrecords", help=".tfrecords filename of validset")
    
    args = parser.parse_args()
    
    return args

args = parse_args()
print("Called with args:", args)

frame_length = int(args.sr*args.frame_size)
stride_length = int(args.sr*args.stride_size)
window_size = args.num_frame*stride_length+frame_length-stride_length

batch_size = 32
num_gpu = 1
global_batch_size = batch_size*num_gpu

assert os.path.splitext(args.trainset_filename)[-1] == ".tfrecords", "extension of trainset_filename must be .tfrecords."
assert os.path.splitext(args.validset_filename)[-1] == ".tfrecords", "extension of validset_filename must be .tfrecords."

if __name__ == "__main__":
    
    def bytes_feature(value: bytes):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def wav_processing(wav_file, tfrecords_file, window_size):
        
        sr, wav_samples = wavfile.read(wav_file)
        if sr != 8000:
            raise ValueError("Sampling rate is expected to be 8kHz!")
            
        assert wav_samples.ndim == 1, "check the size of wav_data"
        num_samples = wav_samples.shape[0]
        if num_samples > window_size:
            num_slices = num_samples//window_size+1
            wav_samples = np.concatenate((wav_samples, wav_samples), axis=0)
            wav_samples = wav_samples[0:window_size*num_slices]
            
            wav_slices = np.reshape(wav_samples, newshape=(num_slices, window_size))
            for wav_slice in wav_slices:
                if np.mean(np.abs(wav_slice)/2**15) < 0.015:
                    num_slices -= 1
                else:
                    wav_bytes = wav_slice.tobytes()
                    example = tf.train.Example(features=tf.train.Features(feature={"wav_raw": bytes_feature(wav_bytes)}))
                    tfrecords_file.write(example.SerializeToString())    
        else:
            num_slices = 1
            while wav_samples.shape[0] < window_size:
                wav_samples = np.concatenate((wav_samples, wav_samples), axis=0)
            
            wav_slice = wav_samples[0:window_size]
            if np.mean( np.abs(wav_slice)/2**15) < 0.015:
                num_slices -= 1
            else:
                wav_bytes = wav_slice.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={"wav_raw": bytes_feature(wav_bytes)}))
                tfrecords_file.write(example.SerializeToString())
        
        return num_slices
    
    ###########################################################################
    wav_files = [os.path.join(args.wav_path, wav) for wav in os.listdir(args.wav_path) if wav.endswith(".wav")]
    num_wav_files = len(wav_files)
    random.shuffle(wav_files)
    num_validset_wav_files = int(args.valid_percent*num_wav_files)
    num_trainset_wav_files = num_wav_files-num_validset_wav_files
    trainset_wav_files = wav_files[0:num_trainset_wav_files]
    validset_wav_files = wav_files[num_trainset_wav_files:num_wav_files]
    
    num_trainset_wav_files = len(trainset_wav_files)
    num_validset_wav_files = len(validset_wav_files)
    
    if not os.path.exists( args.save_path ):
        os.makedirs( args.save_path )
    
    #####################    start processing trainset    #####################
    print("**********  Start processing and writing trainset  **********")
    trainset_tfrecords_filepath = os.path.join(args.save_path, args.trainset_filename)
    
    total_trainset_slices = 0
    begin_time = timeit.default_timer()
    trainset_tfrecords_file = tf.io.TFRecordWriter(trainset_tfrecords_filepath)
    
    for file_count, trainset_wav_file in enumerate(trainset_wav_files):
        print("Processing trainset wav file {}/{} {}{}".format(file_count+1, num_trainset_wav_files, 
                                                               trainset_wav_file, " " * 10), end="\r")
        sys.stdout.flush()
        num_slices = wav_processing(trainset_wav_file, trainset_tfrecords_file, window_size)
        total_trainset_slices += num_slices
    
    print("**************   Post-processing trainset   **************")
    while total_trainset_slices % global_batch_size > 0:
        choose_wav_file = random.choice( trainset_wav_files )
        sr, wav_samples = wavfile.read(choose_wav_file)
        num_samples = wav_samples.shape[0]
        
        if num_samples >= window_size:
            for i in range(0, num_samples, window_size):
                if i+window_size > num_samples:
                    break
                wav_slice = wav_samples[i:i+window_size]
                if np.mean(np.abs(wav_slice)/2**15) > 0.015:
                    wav_bytes = wav_slice.tobytes()
                    example = tf.train.Example(features=tf.train.Features(feature={"wav_raw": bytes_feature(wav_bytes)}))
                    trainset_tfrecords_file.write(example.SerializeToString())
                    
                    total_trainset_slices += 1
                    if total_trainset_slices % global_batch_size == 0:
                        break
    
    trainset_tfrecords_file.close()
    end_time = timeit.default_timer() - begin_time
    print(" ")
    print("*" * 50)
    print("Total processing and writing time: {} s".format(end_time))
    print(" ")
    print(" ")
    
    #####################    start processing validset    #####################
    print("**********  Start processing and writing validset  **********")
    validset_tfrecords_filepath = os.path.join(args.save_path, args.validset_filename)
    
    total_validset_slices = 0
    begin_time = timeit.default_timer()
    validset_tfrecords_file = tf.io.TFRecordWriter(validset_tfrecords_filepath)
    
    for file_count, validset_wav_file in enumerate(validset_wav_files):
        print("Processing validset wav file {}/{} {}{}".format(file_count+1, num_validset_wav_files, 
                                                               validset_wav_file, " " * 10), end="\r")
        sys.stdout.flush()
        num_slices = wav_processing(validset_wav_file, validset_tfrecords_file, window_size)
        total_validset_slices += num_slices
    
    print("**************   Post-processing validset   **************")
    while total_validset_slices % global_batch_size > 0:
        choose_wav_file = random.choice(validset_wav_files)
        sr, wav_samples = wavfile.read(choose_wav_file)
        num_samples = wav_samples.shape[0]
        
        if num_samples >= window_size:  
            for i in range(0, num_samples, window_size):
                if i + window_size > num_samples:
                    break
                wav_slice = wav_samples[i:i+window_size]
                if np.mean(np.abs(wav_slice)/2**15) > 0.015:
                    wav_bytes = wav_slice.tobytes()
                    example = tf.train.Example(features=tf.train.Features(feature={"wav_raw": bytes_feature(wav_bytes)}))
                    validset_tfrecords_file.write(example.SerializeToString())
                    
                    total_validset_slices += 1
                    if total_validset_slices % global_batch_size == 0:
                        break
    
    validset_tfrecords_file.close()
    end_time = timeit.default_timer() - begin_time
    print(" ")
    print("*" * 50)
    print("Total processing and writing time: {} s".format(end_time))
