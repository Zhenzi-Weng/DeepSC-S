"""
Created on Thu Jun 25 23:06:24 2020

@author: zhenzi Weng
"""

import tensorflow as tf

def enframe(wav_signal, length, shift, nf):   # length is the frame length and shift is the frame shift
    # framing
    indices_1 = tf.tile(tf.reshape(tf.range(0, length), [1, length]), [nf, 1])
    indices_2 = tf.transpose(tf.tile(tf.reshape(tf.range(0, nf * shift, shift), [1, nf]), [length, 1]))            
    indices = tf.add(indices_1, indices_2)  # index of each frame
    index = tf.reshape(indices, [nf * length])
    frames_all = tf.zeros([tf.shape(wav_signal)[0], nf, length])
    frames = tf.gather(wav_signal, index, axis = 1) 
    frames_all = tf.reshape(frames, [tf.shape(frames)[0], nf, length], name = "frames_input")

    return frames_all                       


# recover the framed frames into wav signal
def deframe(frames, length, shift, nf): 
    # deframing
    wav1 = tf.reshape(frames[:, 0 : nf - 1, 0 : shift], [tf.shape(frames)[0], (nf - 1) * shift])
    wav2 = tf.reshape(frames[:, nf - 1, 0 : length], [tf.shape(frames)[0], length])
    wav_output = tf.concat([wav1, wav2], axis = 1, name = "wav_out")
            
    return wav_output
    
    
    
   