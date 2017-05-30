# -*- coding: utf-8 -*-

#
# Authors: Dmitriy Serdyuk, Olexa Bilaniuk, Chiheb Trabelsi, Sandeep Subramanian

import keras.backend as K
import keras
from keras.layers import Lambda, add, concatenate, Reshape, Concatenate
from keras.layers.convolutional import (
    Convolution2D, Convolution1D, MaxPooling1D, AveragePooling1D)
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.initializers import Constant
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import Model, Input
from keras.layers.core import Permute

from complexnn import ComplexConv1D, ComplexBN, ComplexDense, GetReal


def get_shallow_convnet(window_size=4096, channels=2, output_size=84):
    inputs = Input(shape=(window_size, channels))

    conv = ComplexConv1D(
        32, 512, strides=16,
        activation='relu')(inputs)
    pool = AveragePooling1D(pool_size=4, strides=2)(conv)

    pool = Permute([2, 1])(pool)
    flattened = Flatten()(pool)

    dense = ComplexDense(2048, activation='relu')(flattened)
    predictions = ComplexDense(
        output_size, 
        activation='sigmoid',
        bias_initializer=Constant(value=-5))(dense)
    predictions = GetReal(predictions)
    model = Model(inputs=inputs, outputs=predictions)

    model.compile(optimizer=Adam(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def get_deep_convnet(window_size=4096, channels=2, output_size=84):
    inputs = Input(shape=(window_size, channels))
    outs = inputs

    outs = (ComplexConv1D(
        16, 6, strides=2, padding='same',
        activation='linear',
        kernel_initializer='complex_independent'))(outs)
    outs = (ComplexBN(axis=-1))(outs)
    outs = (keras.layers.Activation('relu'))(outs)
    outs = (keras.layers.AveragePooling1D(pool_size=2, strides=2))(outs)

    outs = (ComplexConv1D(
        32, 3, strides=2, padding='same',
        activation='linear',
        kernel_initializer='complex_independent'))(outs)
    outs = (ComplexBN(axis=-1))(outs)
    outs = (keras.layers.Activation('relu'))(outs)
    outs = (keras.layers.AveragePooling1D(pool_size=2, strides=2))(outs)
    
    outs = (ComplexConv1D(
        64, 3, strides=1, padding='same',
        activation='linear',
        kernel_initializer='complex_independent'))(outs)
    outs = (ComplexBN(axis=-1))(outs)
    outs = (keras.layers.Activation('relu'))(outs)
    outs = (keras.layers.AveragePooling1D(pool_size=2, strides=2))(outs)

    outs = (ComplexConv1D(
        64, 3, strides=1, padding='same',
        activation='linear',
        kernel_initializer='complex_independent'))(outs)
    outs = (ComplexBN(axis=-1))(outs)
    outs = (keras.layers.Activation('relu'))(outs)
    outs = (keras.layers.AveragePooling1D(pool_size=2, strides=2))(outs)

    outs = (ComplexConv1D(
        128, 3, strides=1, padding='same',
        activation='relu',
        kernel_initializer='complex_independent'))(outs)
    outs = (ComplexConv1D(
        128, 3, strides=1, padding='same',
        activation='linear',
        kernel_initializer='complex_independent'))(outs)
    outs = (ComplexBN(axis=-1))(outs)
    outs = (keras.layers.Activation('relu'))(outs)
    outs = (keras.layers.AveragePooling1D(pool_size=2, strides=2))(outs)

    #outs = (keras.layers.MaxPooling1D(pool_size=2))
    #outs = (Permute([2, 1]))
    outs = (keras.layers.Flatten())(outs)
    outs = (keras.layers.Dense(2048, activation='relu',
                           kernel_initializer='glorot_normal'))(outs)
    predictions = (keras.layers.Dense(output_size, activation='sigmoid',
                                 bias_initializer=keras.initializers.Constant(value=-5)))(outs)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

