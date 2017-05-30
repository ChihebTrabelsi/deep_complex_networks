#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Authors: Chiheb Trabelsi

import numpy as np
from numpy.random import RandomState
import keras.backend as K
from keras import initializers
from keras.initializers import Initializer
from keras.utils.generic_utils import (serialize_keras_object,
                                       deserialize_keras_object)


class IndependentFilters(Initializer):
    # This initialization constructs real-valued kernels
    # that are independent as much as possible from each other
    # while respecting either the He or the Glorot criterion. 
    def __init__(self, kernel_size, input_dim,
                 weight_dim, nb_filters=None,
                 criterion='glorot', seed=None):

        # `weight_dim` is used as a parameter for sanity check
        # as we should not pass an integer as kernel_size when
        # the weight dimension is >= 2.
        # nb_filters == 0 if weights are not convolutional (matrix instead of filters)
        # then in such a case, weight_dim = 2.
        # (in case of 2D input):
        #     nb_filters == None and len(kernel_size) == 2 and_weight_dim == 2
        # conv1D: len(kernel_size) == 1 and weight_dim == 1
        # conv2D: len(kernel_size) == 2 and weight_dim == 2
        # conv3d: len(kernel_size) == 3 and weight_dim == 3

        assert len(kernel_size) == weight_dim and weight_dim in {0, 1, 2, 3}
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.input_dim = input_dim
        self.weight_dim = weight_dim
        self.criterion = criterion
        self.seed = 1337 if seed is None else seed

    def __call__(self, shape, dtype=None):

        if self.nb_filters is not None:
            num_rows = self.nb_filters * self.input_dim
            num_cols = np.prod(self.kernel_size)
        else:
            # in case it is the kernel is a matrix and not a filter
            # which is the case of 2D input (No feature maps).
            num_rows = self.input_dim
            num_cols = self.kernel_size[-1]

        flat_shape = (num_rows, num_cols)
        rng = RandomState(self.seed)
        x = rng.uniform(size=flat_shape)
        u, _, v = np.linalg.svd(x)
        orthogonal_x = np.dot(u, np.dot(np.eye(num_rows, num_cols), v.T))
        if self.nb_filters is not None:
            independent_filters = np.reshape(orthogonal_x, (num_rows,) + tuple(self.kernel_size))
            fan_in, fan_out = initializers._compute_fans(
                tuple(self.kernel_size) + (self.input_dim, self.nb_filters)
            )
        else:
            independent_filters = orthogonal_x
            fan_in, fan_out = (self.input_dim, self.kernel_size[-1])

        if self.criterion == 'glorot':
            desired_var = 2. / (fan_in + fan_out)
        elif self.criterion == 'he':
            desired_var = 2. / fan_in
        else:
            raise ValueError('Invalid criterion: ' + self.criterion)

        multip_constant = np.sqrt (desired_var / np.var(independent_filters))
        scaled_indep = multip_constant * independent_filters

        if self.weight_dim == 2 and self.nb_filters is None:
            weight_real = scaled_real
            weight_imag = scaled_imag
        else:
            kernel_shape = tuple(self.kernel_size) + (self.input_dim, self.nb_filters)
            if self.weight_dim == 1:
                transpose_shape = (1, 0)
            elif self.weight_dim == 2 and self.nb_filters is not None:
                transpose_shape = (1, 2, 0)
            elif self.weight_dim == 3 and self.nb_filters is not None:
                transpose_shape = (1, 2, 3, 0)
            weight = np.transpose(scaled_indep, transpose_shape)
            weight = np.reshape(weight, kernel_shape)

        return weight

    def get_config(self):
        return {'nb_filters': self.nb_filters,
                'kernel_size': self.kernel_size,
                'input_dim': self.input_dim,
                'weight_dim': self.weight_dim,
                'criterion': self.criterion,
                'seed': self.seed}


class ComplexIndependentFilters(Initializer):
    # This initialization constructs complex-valued kernels
    # that are independent as much as possible from each other
    # while respecting either the He or the Glorot criterion.
    def __init__(self, kernel_size, input_dim,
                 weight_dim, nb_filters=None,
                 criterion='glorot', seed=None):

        # `weight_dim` is used as a parameter for sanity check
        # as we should not pass an integer as kernel_size when
        # the weight dimension is >= 2.
        # nb_filters == 0 if weights are not convolutional (matrix instead of filters)
        # then in such a case, weight_dim = 2.
        # (in case of 2D input):
        #     nb_filters == None and len(kernel_size) == 2 and_weight_dim == 2
        # conv1D: len(kernel_size) == 1 and weight_dim == 1
        # conv2D: len(kernel_size) == 2 and weight_dim == 2
        # conv3d: len(kernel_size) == 3 and weight_dim == 3

        assert len(kernel_size) == weight_dim and weight_dim in {0, 1, 2, 3}
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.input_dim = input_dim
        self.weight_dim = weight_dim
        self.criterion = criterion
        self.seed = 1337 if seed is None else seed

    def __call__(self, shape, dtype=None):

        if self.nb_filters is not None:
            num_rows = self.nb_filters * self.input_dim
            num_cols = np.prod(self.kernel_size)
        else:
            # in case it is the kernel is a matrix and not a filter
            # which is the case of 2D input (No feature maps).
            num_rows = self.input_dim
            num_cols = self.kernel_size[-1]

        flat_shape = (int(num_rows), int(num_cols))
        rng = RandomState(self.seed)
        r = rng.uniform(size=flat_shape)
        i = rng.uniform(size=flat_shape)
        z = r + 1j * i
        u, _, v = np.linalg.svd(z)
        unitary_z = np.dot(u, np.dot(np.eye(int(num_rows), int(num_cols)), np.conjugate(v).T))
        real_unitary = unitary_z.real
        imag_unitary = unitary_z.imag
        if self.nb_filters is not None:
            indep_real = np.reshape(real_unitary, (num_rows,) + tuple(self.kernel_size))
            indep_imag = np.reshape(imag_unitary, (num_rows,) + tuple(self.kernel_size))
            fan_in, fan_out = initializers._compute_fans(
                tuple(self.kernel_size) + (int(self.input_dim), self.nb_filters)
            )
        else:
            indep_real = real_unitary
            indep_imag = imag_unitary
            fan_in, fan_out = (int(self.input_dim), self.kernel_size[-1])

        if self.criterion == 'glorot':
            desired_var = 1. / (fan_in + fan_out)
        elif self.criterion == 'he':
            desired_var = 1. / (fan_in)
        else:
            raise ValueError('Invalid criterion: ' + self.criterion)

        multip_real = np.sqrt(desired_var / np.var(indep_real))
        multip_imag = np.sqrt(desired_var / np.var(indep_imag))
        scaled_real = multip_real * indep_real
        scaled_imag = multip_imag * indep_imag

        if self.weight_dim == 2 and self.nb_filters is None:
            weight_real = scaled_real
            weight_imag = scaled_imag
        else:
            kernel_shape = tuple(self.kernel_size) + (int(self.input_dim), self.nb_filters)
            if self.weight_dim == 1:
                transpose_shape = (1, 0)
            elif self.weight_dim == 2 and self.nb_filters is not None:
                transpose_shape = (1, 2, 0)
            elif self.weight_dim == 3 and self.nb_filters is not None:
                transpose_shape = (1, 2, 3, 0)

            weight_real = np.transpose(scaled_real, transpose_shape)
            weight_imag = np.transpose(scaled_imag, transpose_shape)
            weight_real = np.reshape(weight_real, kernel_shape)
            weight_imag = np.reshape(weight_imag, kernel_shape)
        weight = np.concatenate([weight_real, weight_imag], axis=-1)

        return weight

    def get_config(self):
        return {'nb_filters': self.nb_filters,
                'kernel_size': self.kernel_size,
                'input_dim': self.input_dim,
                'weight_dim': self.weight_dim,
                'criterion': self.criterion,
                'seed': self.seed}


class ComplexInit(Initializer):
    # The standard complex initialization using
    # either the He or the Glorot criterion.
    def __init__(self, kernel_size, input_dim,
                 weight_dim, nb_filters=None,
                 criterion='glorot', seed=None):

        # `weight_dim` is used as a parameter for sanity check
        # as we should not pass an integer as kernel_size when
        # the weight dimension is >= 2.
        # nb_filters == 0 if weights are not convolutional (matrix instead of filters)
        # then in such a case, weight_dim = 2.
        # (in case of 2D input):
        #     nb_filters == None and len(kernel_size) == 2 and_weight_dim == 2
        # conv1D: len(kernel_size) == 1 and weight_dim == 1
        # conv2D: len(kernel_size) == 2 and weight_dim == 2
        # conv3d: len(kernel_size) == 3 and weight_dim == 3

        assert len(kernel_size) == weight_dim and weight_dim in {0, 1, 2, 3}
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.input_dim = input_dim
        self.weight_dim = weight_dim
        self.criterion = criterion
        self.seed = 1337 if seed is None else seed

    def __call__(self, shape, dtype=None):

        if self.nb_filters is not None:
            kernel_shape = tuple(self.kernel_size) + (int(self.input_dim), self.nb_filters)
        else:
            kernel_shape = (int(self.input_dim), self.kernel_size[-1])

        fan_in, fan_out = initializers._compute_fans(
            tuple(self.kernel_size) + (self.input_dim, self.nb_filters)
        )

        if self.criterion == 'glorot':
            s = 1. / (fan_in + fan_out)
        elif self.criterion == 'he':
            s = 1. / fan_in
        else:
            raise ValueError('Invalid criterion: ' + self.criterion)
        rng = RandomState(self.seed)
        modulus = rng.rayleigh(scale=s, size=kernel_shape)
        phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
        weight_real = modulus * np.cos(phase)
        weight_imag = modulus * np.sin(phase)
        weight = np.concatenate([weight_real, weight_imag], axis=-1)

        return weight


class SqrtInit(Initializer):
    def __call__(self, shape, dtype=None):
        return K.constant(1 / K.sqrt(2), shape=shape, dtype=dtype)


# Aliases:
sqrt_init = SqrtInit
independent_filters = IndependentFilters
complex_init = ComplexInit