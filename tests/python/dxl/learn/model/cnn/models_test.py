# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy
import os 

from dxl.learn.model.cnn import Conv2D, StackedConv2D, InceptionBlock

# Conv2D
x = numpy.random.randint(0, 255, [1, 100, 100, 3])
input_tensor = tf.constant(x, dtype="float32")

n = Conv2D(
    name='Conv2D_test',
    input_tensor=input_tensor,
    filters=32,
    kernel_size=[5,5],
    strides=(2, 2),
    padding='same',
    activation='basic'
)
shape = n.outputs['main'].shape
if (shape[1] != 50) or (shape[2] != 50) or (shape[3] != 32):
    print("Conv2D test failed!!!")
    raise Exception("shape dont match")
else:
    print("Conv2D pass test!!!")

# StackedConv2D
x = numpy.random.randint(0, 255, [1, 100, 100, 3])
input_tensor = tf.constant(x, dtype="float32")

n = StackedConv2D(
    name='StackedConv2D_test',
    input_tensor=input_tensor,
    nb_layers=2,
    filters=32,
    kernel_size=[5,5],
    strides=(2, 2),
    padding='same',
    activation='basic'
)
shape = n.outputs['main'].shape
if (shape[1] != 25) or (shape[2] != 25) or (shape[3] != 32):
    print("StackedConv2D test failed!!!")
    raise Exception("shape dont match")
else:
    print("StackedConv2D pass test!!!")

# InceptionBlock
x = numpy.random.randint(0, 255, [2, 100, 100, 3])
input_tensor = tf.constant(x, dtype="float32")

n =  InceptionBlock(
    name='InceptionBlock_test',
    input_tensor=input_tensor,
    paths=3,
    activation='incept'
)
shape = n.outputs['main'].shape
if (shape[1] != 100) or (shape[2] != 100) or (shape[3] != 3):
    print("InceptionBlock test failed!!!")
    raise Exception("shape dont match")
else:
    print("InceptionBlock pass test!!!")
