"""
Created on Thu Jun 25 23:06:24 2020

@author: zhenzi Weng
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def wav_norm(wav_input):
    
    batch_mean, batch_var = tf.nn.moments(wav_input, axes=[-1])
    batch_mean = tf.expand_dims(batch_mean, axis=-1)
    batch_var = tf.expand_dims(batch_var, axis=-1)
    
    wav_input_norm = tf.math.divide( tf.math.subtract(wav_input, batch_mean), tf.math.sqrt(batch_var) )
    
    return wav_input_norm, batch_mean, batch_var

def wav_denorm(wav_output, batch_mean, batch_var):
    
    wav_output_denorm = tf.math.add( tf.math.multiply(wav_output, tf.math.sqrt(batch_var)), batch_mean )
    
    return wav_output_denorm

def enframe(wav_input, num_frame, frame_length, stride_length):

    indices_1 = tf.tile(tf.reshape(tf.range(0, frame_length), [1, frame_length]), [num_frame, 1])
    indices_2 = tf.transpose(tf.tile(tf.reshape(tf.range(0, num_frame * stride_length, stride_length), [1, num_frame]), [frame_length, 1]))            
    indices = tf.math.add(indices_1, indices_2)  # index of each frame
    
    index = tf.reshape(indices, [num_frame * frame_length])
    frame_input = tf.gather(wav_input, index, axis = 1)
    frame_input = tf.reshape(frame_input, [tf.shape(frame_input)[0], num_frame, frame_length])

    return frame_input                   

def deframe(frame_output, num_frame, frame_length, stride_length): 

    wav1 = tf.reshape(frame_output[:, 0 : num_frame - 1, 0 : stride_length], 
                      [tf.shape(frame_output)[0], (num_frame - 1) * stride_length])
    wav2 = tf.reshape(frame_output[:, num_frame - 1, 0 : frame_length], 
                      [tf.shape(frame_output)[0], frame_length])
    wav_output = tf.concat([wav1, wav2], axis = 1)
            
    return wav_output
