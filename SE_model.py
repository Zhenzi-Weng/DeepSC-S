# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 23:56:14 2020

@author: Zhenzi Weng
"""
import tensorflow as tf

from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.framework import arg_scope

depth = 16
cardinality = 2
reduction_ratio = 4



def Global_Average_Pooling(x):
    return global_avg_pool(x, name="Global_avg_pooling")


def Average_pooling(x, pool_size=[2,2], stride=2, padding="SAME"):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def conv_layer(inputs, filters, kernel, stride, layer_name = "conv"):
    
    with tf.name_scope(layer_name):
    # convolution
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
        bias_initializer = tf.constant_initializer(value=0.0)
        
        conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel, strides=stride,
                                      kernel_initializer=kernel_initializer, 
                                      bias_initializer = bias_initializer,
                                      padding="SAME")(inputs)
        
        return conv


def Batch_Normalization(conv, training, scope):
    # batch normalization
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=conv, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=conv, is_training=training, reuse=True))


def Fully_connected(x, units, layer_name="fully_connected") :
    with tf.name_scope(layer_name) :
        return tf.keras.layers.Dense(units=units, use_bias=False)(x)
    


class SE_ResNeXt():
    def __init__(self, inputs, filters, out_dim, is_train, name, blocks):
        
        self.training = is_train
        self.model = self.Build_SEnet(inputs, filters, out_dim, name, blocks)
        
    def first_layer(self, inputs, filters, scope):
        with tf.name_scope(scope):
            conv = conv_layer(inputs, filters, kernel=[5, 5], stride=[1, 1], layer_name = scope + "_conv1")
            conv_BN = Batch_Normalization(conv, training = self.training, scope = scope + "_BN1")
            conv_out = tf.nn.relu(conv_BN)
        
            return conv_out


    def transform_layer(self, inputs, stride, scope):
        with tf.name_scope(scope):
            conv1 = conv_layer(inputs, filters=depth, kernel=[5,5], stride=stride, layer_name=scope+"_conv1")
            conv1_BN = Batch_Normalization(conv1, training=self.training, scope=scope+"_BN1")
            conv1_out = tf.nn.relu(conv1_BN)
            
            return conv1_out


    def split_layer(self, fir_out, stride, layer_name):
        with tf.name_scope(layer_name):
            layers_split = list()
            for i in range(cardinality):
                splits = self.transform_layer(fir_out, stride, scope=layer_name + "_splitN_" + str(i))
                layers_split.append(splits)
                
            split_out = tf.concat(layers_split, axis = 3)
            
            return split_out
        
        
    def transition_layer(self, split_out, out_dim, scope):
        with tf.name_scope(scope):
            conv1 = conv_layer(split_out, filters=out_dim, kernel=[1,1], stride=[1, 1], layer_name=scope+"_conv1")
            conv1_BN = Batch_Normalization(conv1, training=self.training, scope=scope+"_batch1")
            # conv1_out = tf.nn.leaky_relu(conv1_BN)
            
            return conv1_BN


    def squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name) :


            squeeze = Global_Average_Pooling(input_x)

            excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+"_fully_connected1")
            excitation = tf.nn.relu(excitation)
            excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+"_fully_connected2")
            excitation = tf.nn.sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1,1,1,out_dim])
            scale = input_x * excitation

            return scale     

    def residual_layer(self, input_res, out_dim, layer_num, res_block):
            for i in range(res_block):
                stride = 1
    
                x = self.split_layer(input_res, stride=stride, layer_name="split_layer_"+layer_num+"_"+str(i))
                x = self.transition_layer(x, out_dim=out_dim, scope="trans_layer_"+layer_num+"_"+str(i))
                x = self.squeeze_excitation_layer(x, out_dim=out_dim, ratio=reduction_ratio, layer_name="squeeze_layer_"+layer_num+"_"+str(i))
    
                pad_input_res = input_res
    
                input_res = tf.nn.relu(x + pad_input_res)
    
            return input_res


    def Build_SEnet(self, inputs, filters, out_dim, name, blocks):
            # first layer
            fir_out = self.first_layer(inputs, filters, scope = name + "first_layer")
            # residual layer
            res_1 = self.residual_layer(fir_out, out_dim=out_dim, layer_num = name + "residual_1", res_block=blocks)
            
            return res_1

