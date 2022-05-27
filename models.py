# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 13:36:35 2022

@author: Zhenzi Weng
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, GlobalAveragePooling2D, Dense, Concatenate, BatchNormalization
from speech_processing import enframe, deframe, wav_norm, wav_denorm

def conv_bn_layer(inputs, filters, strides, name):
    
    conv = Conv2D(filters=filters, kernel_size=(5, 5), strides=strides,
                  padding="same", use_bias=False, name="{}_conv".format(name))(inputs)
    conv_bn = BatchNormalization(name="{}_bn".format(name))(conv)
    
    return conv_bn

def convtrans_bn_layer(inputs, filters, strides, name):
    
    convtrans = Conv2DTranspose(filters=filters, kernel_size=(5, 5), strides=strides,
                                padding="same", use_bias=False, name="{}_convtrans".format(name))(inputs)
    convtrans_bn = BatchNormalization(name="{}_bn".format(name))(convtrans)
    
    return convtrans_bn

###################  SE-ResNet  ###################
depth = 128
cardinality = 4
reduction_ratio = 4

def global_average_pooling(inputs, name):
    
    pooling_output = GlobalAveragePooling2D(name="{}_squeeze".format(name))(inputs) 
    
    return pooling_output

def transform_layer(inputs, filters, strides, name):
   
    conv_bn = conv_bn_layer(inputs=inputs, filters=filters, 
                            strides=strides, name=name)
    transform_output = tf.nn.relu(conv_bn)
    
    return transform_output

def split_layer(inputs, filters, strides, name):
    
    layers_split = list()
    for i in range(cardinality):
        splits = transform_layer(inputs=inputs, filters=filters,
                                 strides=strides, name="{}_transform{}".format(name, i))
        layers_split.append(splits)
    split_output = Concatenate(axis=-1)(layers_split)
    
    return split_output

def transition_layer(inputs, out_dim, name):
    
    transition_output = Conv2D(filters=out_dim, kernel_size=(1, 1), strides=(1, 1),
                               padding="same", use_bias=False, name="{}_conv".format(name))(inputs)
    transition_output = BatchNormalization(name="{}_bn".format(name))(transition_output)
    
    return transition_output

def SE_layer(SE_input, out_dim, reduction_ratio, name):
    
    squeeze = global_average_pooling(SE_input, name=name)
    excitation = Dense(units=out_dim/reduction_ratio, use_bias=False, 
                       name="{}_dense1".format(name))(squeeze)
    excitation = tf.nn.relu(excitation)
    excitation = Dense(units=out_dim, use_bias=False,
                       name="{}_dense2".format(name))(excitation)
    excitation = tf.keras.activations.sigmoid(excitation)
    
    SE_output = tf.reshape(excitation, [-1, 1, 1, out_dim])
    
    return SE_output

def SEResNet(inputs, out_dim, name):
    
    split_output = split_layer(inputs, filters=depth,
                               strides=(1, 1), name="{}_split".format(name))
    transition_output = transition_layer(split_output, out_dim=out_dim,
                                         name="{}_transition".format(name))
    SE_output = SE_layer(transition_output, out_dim=out_dim, 
                         reduction_ratio=reduction_ratio, name="{}_SE".format(name))
    
    SEResNet_output = tf.math.add(inputs, tf.math.multiply(SE_output, transition_output))
    
    return SEResNet_output

###################  model function  ###################
# semantic encoder
class Sem_Enc(object):  
    
    def __init__(self, frame_length, stride_length, args):
        
        self.num_frame = args.num_frame
        self.frame_length = frame_length
        self.stride_length = stride_length
        self.sem_enc_outdims = args.sem_enc_outdims
        
    def __call__(self, _input):
        
        # preprocessing _intput
        _input, batch_mean, batch_var = wav_norm(_input)
        _input = enframe(_input, self.num_frame, self.frame_length, self.stride_length)
        _input = tf.expand_dims(_input, axis=-1)
        
        ######################   semantic encoder   ######################
        _output = conv_bn_layer(_input, filters=self.sem_enc_outdims[0],
                                strides=(2, 2), name="sem_enc_cnn1")
        _output = tf.nn.relu(_output)
        _output = conv_bn_layer(_output, filters=self.sem_enc_outdims[1],
                                strides=(2, 2), name="sem_enc_cnn2")
        _output = tf.nn.relu(_output)
        for module_count, outdim in enumerate(self.sem_enc_outdims[2:]):
            module_id = module_count + 1
            _output = SEResNet(_output, out_dim=outdim,
                               name="sem_enc_module{}".format(module_id))
            _output = tf.nn.relu(_output)
        
        return _output, batch_mean, batch_var

# channel encoder
class Chan_Enc(object):  
    
    def __init__(self, frame_length, args):
        
        self.num_frame = args.num_frame
        self.frame_length = frame_length
        self.chan_enc_filters = args.chan_enc_filters
        
    def __call__(self, _intput):
        
        ######################   chanel encoder   ######################         
        _output = conv_bn_layer(_intput, filters=self.chan_enc_filters[0],
                                strides=(1, 1), name="chan_enc_cnn1")
        
        return _output

# channel model
class Chan_Model(object):  
    """Define MIMO channel model."""
    def __init__(self, name):
        
        self.name = name
            
    def __call__(self, _input, std):
        
        _input = tf.transpose(_input, perm=[0, 3, 1, 2])
        
        batch_size = tf.shape(_input)[0]
        _shape = _input.get_shape().as_list()
        assert (_shape[2]*_shape[3]) % 2 == 0, "number of transmitted symbols must be an integer."
        
        # reshape layer and normalize the average power of each dim in x into 0.5
        x = tf.reshape(_input, [batch_size, _shape[1], _shape[2]*_shape[3]//2, 2])
        x_norm = tf.math.sqrt(_shape[2]*_shape[3]//2 / 2.0) * tf.math.l2_normalize(x, axis=2)
        
        x_real = x_norm[:, :, :, 0]
        x_imag = x_norm[:, :, :, 1]
        x_complex = tf.dtypes.complex(real=x_real, imag=x_imag)
        
        # channel h
        h = tf.random.normal(shape=[batch_size, _shape[1], 1, 2], dtype=tf.float32)
        h = (tf.math.sqrt(1./2.) + tf.math.sqrt(1./2.)*h) / tf.math.sqrt(2.)
        h_real = h[:, :, :, 0]
        h_imag = h[:, :, :, 1]
        h_complex = tf.dtypes.complex(real=h_real, imag=h_imag)
        
        # noise n
        n = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=std, dtype=tf.float32)
        n_real = n[:, :, :, 0]
        n_imag = n[:, :, :, 1]
        n_complex = tf.dtypes.complex(real=n_real, imag=n_imag)
        
        # receive y
        y_complex = tf.math.multiply(h_complex, x_complex) + n_complex
        
        # estimate x_hat with perfect CSI
        x_hat_complex = tf.math.divide(y_complex, h_complex)
        
        # convert complex to real
        x_hat_real = tf.expand_dims(tf.math.real(x_hat_complex), axis=-1)
        x_hat_imag = tf.expand_dims(tf.math.imag(x_hat_complex), axis=-1)
        x_hat = tf.concat([x_hat_real, x_hat_imag], -1)
        
        _output = tf.reshape(x_hat, shape=tf.shape(_input))
        _output = tf.transpose(_output, perm=[0, 2, 3, 1])
        
        return _output

# channel decoder
class Chan_Dec(object):  
    
    def __init__(self, frame_length, args):
        
        self.num_frame = args.num_frame
        self.frame_length = frame_length
        self.chan_dec_filters = args.chan_dec_filters
        
    def __call__(self, _input):
        
        ######################   channel encoder   ######################         
        _output = conv_bn_layer(_input, filters=self.chan_dec_filters[0],
                                strides=(1, 1), name="chan_dec_cnn1")
        _output = tf.nn.relu(_output)
        
        return _output

# semantic decoder
class Sem_Dec(object):  
    
    def __init__(self, frame_length, stride_length, args):
        
        self.num_frame = args.num_frame
        self.frame_length=frame_length
        self.stride_length = stride_length
        self.sem_dec_outdims = args.sem_dec_outdims
        
    def __call__(self, _input, batch_mean, batch_var):
        
        ######################   semantic decoder   ######################
        for module_count, outdim in enumerate(self.sem_dec_outdims[:-2]):
            module_id = module_count + 1
            _input = SEResNet(_input, out_dim=outdim,
                              name="sem_dec_module{}".format(module_id))
            _input = tf.nn.relu(_input)
        _output = convtrans_bn_layer(_input, filters=self.sem_dec_outdims[-2],
                                     strides=(2, 2), name="sem_dec_cnn1")
        _output = tf.nn.relu(_output)
        _output = convtrans_bn_layer(_output, filters=self.sem_dec_outdims[-1],
                                     strides=(2, 2), name="sem_dec_cnn2")
        _output = tf.nn.relu(_output)
        
        # last layer
        _output = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1),
                         padding="same", use_bias=False, name="sem_dec_cnn3")(_output)
        _output = tf.squeeze(_output, axis=-1)
        
        # processing _output
        _output = deframe(_output, self.num_frame, self.frame_length, self.stride_length)
        _output = wav_denorm(_output, batch_mean, batch_var)
        
        return _output

###################  defined models  ###################
# semantic encoder
def sem_enc_model(frame_length, stride_length, args):
    
    wav_size = args.num_frame*stride_length+frame_length-stride_length
    _input = tf.keras.layers.Input(name="wav_input", shape=(wav_size,), dtype=tf.float32)
    
    sem_enc = Sem_Enc(frame_length, stride_length, args)
    _output, batch_mean, batch_var = sem_enc(_input)
    
    model = tf.keras.models.Model(inputs=_input,
                                  outputs=[_output, batch_mean, batch_var],
                                  name="Semantic_Encoder")
    
    return model
    
# channel encoder
def chan_enc_model(frame_length, args):

    _input = tf.keras.layers.Input(name="chan_enc_input",
                                   shape=(args.num_frame//4, frame_length//4, args.sem_enc_outdims[-1]),
                                   dtype=tf.float32)
    
    chan_enc = Chan_Enc(frame_length, args)
    _output = chan_enc(_input)
    
    model = tf.keras.models.Model(inputs=_input, outputs=_output, name="Channel_Encoder")
    
    return model

# channel decoder
def chan_dec_model(frame_length, args):

    _input = tf.keras.layers.Input(name="chan_dec_input",
                                   shape=(args.num_frame//4, frame_length//4, args.chan_enc_filters[-1]),
                                   dtype=tf.float32)
    
    chan_dec = Chan_Dec(frame_length, args)
    _output = chan_dec(_input)
    
    model = tf.keras.models.Model(inputs=_input, outputs=_output, name="Channel_Decoder")
    
    return model

# semantic decoder
def sem_dec_model(frame_length, stride_length, args):
    
    _intput = tf.keras.layers.Input(name="sem_dec_intput",
                                    shape=(args.num_frame//4, frame_length//4, args.chan_dec_filters[-1]),
                                    dtype=tf.float32)
    batch_mean = tf.keras.layers.Input(name="batch_mean", shape=(1,), dtype=tf.float32)
    batch_var = tf.keras.layers.Input(name="batch_var", shape=(1,), dtype=tf.float32)
    
    sem_dec = Sem_Dec(frame_length, stride_length, args)
    _output = sem_dec(_intput, batch_mean, batch_var)
    
    model = tf.keras.models.Model(inputs=[_intput, batch_mean, batch_var],
                                  outputs=_output,
                                  name="Semantic_Decoder")
    
    return model
