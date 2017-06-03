# -*- coding: utf-8 -*-

#
# Authors: Dmitriy Serdyuk

import keras
from keras.models import Model
from keras.layers import (
    Input, BatchNormalization, Activation, Dense, Flatten)
from keras.layers.convolutional import Convolution1D, AveragePooling1D
from keras.regularizers import l2


def get_mlp(window_size=4096, output_size=84):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(window_size, 1)))
    model.add(keras.layers.Dense(2048, activation='relu',
                                 kernel_initializer='glorot_normal'))
    model.add(keras.layers.Dense(output_size, activation='sigmoid',
                                 kernel_initializer='glorot_normal',
                                 bias_initializer=keras.initializers.Constant(value=-5)))
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def get_shallow_convnet(window_size=4096, channels=1, output_size=84):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(
        64, 512, strides=16, input_shape=(window_size, channels),
        activation='relu',
        kernel_initializer='glorot_normal'))
    model.add(keras.layers.MaxPooling1D(pool_size=4, strides=2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(2048, activation='relu',
                                 kernel_initializer='glorot_normal'))
    model.add(keras.layers.Dense(output_size, activation='sigmoid',
                                 bias_initializer=keras.initializers.Constant(value=-5)))
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def get_deep_convnet(window_size=4096, channels=1, output_size=84):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(
        64, 7, strides=3, input_shape=(window_size, channels),
        activation='linear',
        kernel_initializer='glorot_normal'))
    model.add(keras.layers.normalization.BatchNormalization(axis=-1))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))

    model.add(keras.layers.Conv1D(
        64, 3, strides=2,
        activation='linear',
        kernel_initializer='glorot_normal'))
    model.add(keras.layers.normalization.BatchNormalization(axis=-1))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))
    
    model.add(keras.layers.Conv1D(
        128, 3, strides=1,
        activation='linear',
        kernel_initializer='glorot_normal'))
    model.add(keras.layers.normalization.BatchNormalization(axis=-1))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))

    model.add(keras.layers.Conv1D(
        128, 3, strides=1,
        activation='linear',
        kernel_initializer='glorot_normal'))
    model.add(keras.layers.normalization.BatchNormalization(axis=-1))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))

    model.add(keras.layers.Conv1D(
        256, 3, strides=1,
        activation='relu',
        kernel_initializer='glorot_normal'))
    model.add(keras.layers.Conv1D(
        256, 3, strides=1,
        activation='linear',
        kernel_initializer='glorot_normal'))
    model.add(keras.layers.normalization.BatchNormalization(axis=-1))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(2048, activation='relu',
                                 kernel_initializer='glorot_normal'))
    model.add(keras.layers.Dense(output_size, activation='sigmoid',
                                 bias_initializer=keras.initializers.Constant(value=-5)))
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

