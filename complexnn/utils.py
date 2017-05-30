#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Authors: Dmitriy Serdyuk, Olexa Bilaniuk, Chiheb Trabelsi

import keras.backend as K
from keras.layers import Lambda

#
# GetReal/GetImag Lambda layer Implementation
#


def get_realpart(x):
    image_format = K.image_data_format()
    ndim = K.ndim(x)
    input_shape = K.shape(x)

    if (image_format == 'channels_first' and ndim != 3) or ndim == 2:
        input_dim = input_shape[1] // 2
        return x[:, :input_dim]

    input_dim = input_shape[-1] // 2
    if ndim == 3:
        return x[:, :, :input_dim]
    elif ndim == 4:
        return x[:, :, :, :input_dim]
    elif ndim == 5:
        return x[:, :, :, :, :input_dim]


def get_imagpart(x):
    image_format = K.image_data_format()
    ndim = K.ndim(x)
    input_shape = K.shape(x)

    if (image_format == 'channels_first' and ndim != 3) or ndim == 2:
        input_dim = input_shape[1] // 2
        return x[:, input_dim:]

    input_dim = input_shape[-1] // 2
    if ndim == 3:
        return x[:, :, input_dim:]
    elif ndim == 4:
        return x[:, :, :, input_dim:]
    elif ndim == 5:
        return x[:, :, :, :, input_dim:]


def get_abs(x):
    real = get_realpart(x)
    imag = get_imagpart(x)

    return K.sqrt(real * real + imag * imag)


def getpart_output_shape(input_shape):
    returned_shape = list(input_shape[:])
    image_format = K.image_data_format()
    ndim = len(returned_shape)

    if (image_format == 'channels_first' and ndim != 3) or ndim == 2:
        axis = 1
    else:
        axis = -1

    returned_shape[axis] = returned_shape[axis] // 2

    return tuple(returned_shape)

GetReal = Lambda(get_realpart, output_shape=getpart_output_shape)
GetImag = Lambda(get_imagpart, output_shape=getpart_output_shape)
GetAbs = Lambda(get_abs, output_shape=getpart_output_shape)
