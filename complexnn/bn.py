#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Authors: Chiheb Trabelsi, Olexa Bilaniuk
#
# Note: The implementation of complex Batchnorm is based on
#       the Keras implementation of batch Normalization
#       available here:
#       https://github.com/fchollet/keras/blob/master/keras/layers/normalization.py

import numpy as np
from keras.layers import Layer, InputSpec
from keras import initializers, regularizers, constraints
import keras.backend as K


def sqrt_init(shape, dtype=None):
    value = (1 / K.sqrt(2)) * K.ones(shape)
    return value


def complex_standardization(input_centred, Vrr, Vii, Vri,
                            layernorm=False, axis=-1):
    
    ndim = K.ndim(input_centred)
    input_dim = K.shape(input_centred)[axis] // 2
    variances_broadcast = [1] * ndim
    variances_broadcast[axis] = input_dim
    if layernorm:
        variances_broadcast[0] = K.shape(input_centred)[0]

    # We require the covariance matrix's inverse square root. That first requires
    # square rooting, followed by inversion (I do this in that order because during
    # the computation of square root we compute the determinant we'll need for
    # inversion as well).

    # tau = Vrr + Vii = Trace. Guaranteed >= 0 because SPD
    tau = Vrr + Vii
    # delta = (Vrr * Vii) - (Vri ** 2) = Determinant. Guaranteed >= 0 because SPD
    delta = (Vrr * Vii) - (Vri ** 2)

    s = np.sqrt(delta) # Determinant of square root matrix
    t = np.sqrt(tau + 2 * s)

    # The square root matrix could now be explicitly formed as
    #       [ Vrr+s Vri   ]
    # (1/t) [ Vir   Vii+s ]
    # https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
    # but we don't need to do this immediately since we can also simultaneously
    # invert. We can do this because we've already computed the determinant of
    # the square root matrix, and can thus invert it using the analytical
    # solution for 2x2 matrices
    #      [ A B ]             [  D  -B ]
    # inv( [ C D ] ) = (1/det) [ -C   A ]
    # http://mathworld.wolfram.com/MatrixInverse.html
    # Thus giving us
    #           [  Vii+s  -Vri   ]
    # (1/s)(1/t)[ -Vir     Vrr+s ]
    # So we proceed as follows:

    inverse_st = 1.0 / (s * t)
    Wrr = (Vii + s) * inverse_st
    Wii = (Vrr + s) * inverse_st
    Wri = -Vri * inverse_st

    # And we have computed the inverse square root matrix W = sqrt(V)!
    # Normalization. We multiply, x_normalized = W.x.

    # The returned result will be a complex standardized input
    # where the real and imaginary parts are obtained as follows:
    # x_real_normed = Wrr * x_real_centred + Wri * x_imag_centred
    # x_imag_normed = Wri * x_real_centred + Wii * x_imag_centred

    broadcast_Wrr = K.reshape(Wrr, variances_broadcast)
    broadcast_Wri = K.reshape(Wri, variances_broadcast)
    broadcast_Wii = K.reshape(Wii, variances_broadcast)

    cat_W_4_real = K.concatenate([broadcast_Wrr, broadcast_Wii], axis=axis)
    cat_W_4_imag = K.concatenate([broadcast_Wri, broadcast_Wri], axis=axis)

    if (axis == 1 and ndim != 3) or ndim == 2:
        centred_real = input_centred[:, :input_dim]
        centred_imag = input_centred[:, input_dim:]
    elif ndim == 3:
        centred_real = input_centred[:, :, :input_dim]
        centred_imag = input_centred[:, :, input_dim:]
    elif axis == -1 and ndim == 4:
        centred_real = input_centred[:, :, :, :input_dim]
        centred_imag = input_centred[:, :, :, input_dim:]
    elif axis == -1 and ndim == 5:
        centred_real = input_centred[:, :, :, :, :input_dim]
        centred_imag = input_centred[:, :, :, :, input_dim:]
    else:
        raise ValueError(
            'Incorrect Batchnorm combination of axis and dimensions. axis should be either 1 or -1. '
            'axis: ' + str(self.axis) + '; ndim: ' + str(ndim) + '.'
        )
    rolled_input = K.concatenate([centred_imag, centred_real], axis=axis)

    output = cat_W_4_real * input_centred + cat_W_4_imag * rolled_input

    #   Wrr * x_real_centered | Wii * x_imag_centered
    # + Wri * x_imag_centered | Wri * x_real_centered
    # -----------------------------------------------
    # = output

    return output


def ComplexBN(input_centred, Vrr, Vii, Vri, beta,
               gamma_rr, gamma_ri, gamma_ii, scale=True,
               center=True, layernorm=False, axis=-1):

    ndim = K.ndim(input_centred)
    input_dim = K.shape(input_centred)[axis] // 2
    if scale:
        gamma_broadcast_shape = [1] * ndim
        gamma_broadcast_shape[axis] = input_dim
    if center:
        broadcast_beta_shape = [1] * ndim
        broadcast_beta_shape[axis] = input_dim * 2

    if scale:
        standardized_output = complex_standardization(
            input_centred, Vrr, Vii, Vri,
            layernorm,
            axis=axis
        )

        # Now we perform th scaling and Shifting of the normalized x using
        # the scaling parameter
        #           [  gamma_rr gamma_ri  ]
        #   Gamma = [  gamma_ri gamma_ii  ]
        # and the shifting parameter
        #    Beta = [beta_real beta_imag].T
        # where:
        # x_real_BN = gamma_rr * x_real_normed + gamma_ri * x_imag_normed + beta_real
        # x_imag_BN = gamma_ri * x_real_normed + gamma_ii * x_imag_normed + beta_imag
        
        broadcast_gamma_rr = K.reshape(gamma_rr, gamma_broadcast_shape)
        broadcast_gamma_ri = K.reshape(gamma_ri, gamma_broadcast_shape)
        broadcast_gamma_ii = K.reshape(gamma_ii, gamma_broadcast_shape)

        cat_gamma_4_real = K.concatenate([broadcast_gamma_rr, broadcast_gamma_ii], axis=axis)
        cat_gamma_4_imag = K.concatenate([broadcast_gamma_ri, broadcast_gamma_ri], axis=axis)
        if (axis == 1 and ndim != 3) or ndim == 2:
            centred_real = standardized_output[:, :input_dim]
            centred_imag = standardized_output[:, input_dim:]
        elif ndim == 3:
            centred_real = standardized_output[:, :, :input_dim]
            centred_imag = standardized_output[:, :, input_dim:]
        elif axis == -1 and ndim == 4:
            centred_real = standardized_output[:, :, :, :input_dim]
            centred_imag = standardized_output[:, :, :, input_dim:]
        elif axis == -1 and ndim == 5:
            centred_real = standardized_output[:, :, :, :, :input_dim]
            centred_imag = standardized_output[:, :, :, :, input_dim:]
        else:
            raise ValueError(
                'Incorrect Batchnorm combination of axis and dimensions. axis should be either 1 or -1. '
                'axis: ' + str(self.axis) + '; ndim: ' + str(ndim) + '.'
            )
        rolled_standardized_output = K.concatenate([centred_imag, centred_real], axis=axis)
        if center:
            broadcast_beta = K.reshape(beta, broadcast_beta_shape)
            return cat_gamma_4_real * standardized_output + cat_gamma_4_imag * rolled_standardized_output + broadcast_beta
        else:
            return cat_gamma_4_real * standardized_output + cat_gamma_4_imag * rolled_standardized_output
    else:
        if center:
            broadcast_beta = K.reshape(beta, broadcast_beta_shape)
            return input_centred + broadcast_beta
        else:
            return input_centred


class ComplexBatchNormalization(Layer):
    """Complex version of the real domain 
    Batch normalization layer (Ioffe and Szegedy, 2014).
    Normalize the activations of the previous complex layer at each batch,
    i.e. applies a transformation that maintains the mean of a complex unit
    close to the null vector, the 2 by 2 covariance matrix of a complex unit close to identity
    and the 2 by 2 relation matrix, also called pseudo-covariance, close to the 
    null matrix.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=2` in `ComplexBatchNormalization`.
        momentum: Momentum for the moving statistics related to the real and
            imaginary parts.
        epsilon: Small float added to each of the variances related to the
            real and imaginary parts in order to avoid dividing by zero.
        center: If True, add offset of `beta` to complex normalized tensor.
            If False, `beta` is ignored.
            (beta is formed by real_beta and imag_beta)
        scale: If True, multiply by the `gamma` matrix.
            If False, `gamma` is not used.
        beta_initializer: Initializer for the real_beta and the imag_beta weight.
        gamma_diag_initializer: Initializer for the diagonal elements of the gamma matrix.
            which are the variances of the real part and the imaginary part.
        gamma_off_initializer: Initializer for the off-diagonal elements of the gamma matrix.
        moving_mean_initializer: Initializer for the moving means.
        moving_variance_initializer: Initializer for the moving variances.
        moving_covariance_initializer: Initializer for the moving covariance of
            the real and imaginary parts.
        beta_regularizer: Optional regularizer for the beta weights.
        gamma_regularizer: Optional regularizer for the gamma weights.
        beta_constraint: Optional constraint for the beta weights.
        gamma_constraint: Optional constraint for the gamma weights.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
    """

    def __init__(self,
                 axis=-1,
                 momentum=0.9,
                 epsilon=1e-4,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_diag_initializer='sqrt_init',
                 gamma_off_initializer='zeros',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='sqrt_init',
                 moving_covariance_initializer='zeros',
                 beta_regularizer=None,
                 gamma_diag_regularizer=None,
                 gamma_off_regularizer=None,
                 beta_constraint=None,
                 gamma_diag_constraint=None,
                 gamma_off_constraint=None,
                 **kwargs):
        super(ComplexBatchNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        if gamma_diag_initializer != 'sqrt_init':
            self.gamma_diag_initializer = initializers.get(gamma_diag_initializer)
        else:
            self.gamma_diag_initializer = sqrt_init
        self.gamma_off_initializer = initializers.get(gamma_off_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        if moving_variance_initializer != 'sqrt_init':
            self.moving_variance_initializer = initializers.get(moving_variance_initializer)
        else:
            self.moving_variance_initializer = sqrt_init
        self.moving_covariance_initializer = initializers.get(moving_covariance_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_diag_regularizer = regularizers.get(gamma_diag_regularizer)
        self.gamma_off_regularizer = regularizers.get(gamma_off_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_diag_constraint = constraints.get(gamma_diag_constraint)
        self.gamma_off_constraint = constraints.get(gamma_off_constraint)

    def build(self, input_shape):

        ndim = len(input_shape)

        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})

        param_shape = (input_shape[self.axis] // 2,)

        if self.scale:
            self.gamma_rr = self.add_weight(shape=param_shape,
                                            name='gamma_rr',
                                            initializer=self.gamma_diag_initializer,
                                            regularizer=self.gamma_diag_regularizer,
                                            constraint=self.gamma_diag_constraint)
            self.gamma_ii = self.add_weight(shape=param_shape,
                                            name='gamma_ii',
                                            initializer=self.gamma_diag_initializer,
                                            regularizer=self.gamma_diag_regularizer,
                                            constraint=self.gamma_diag_constraint)
            self.gamma_ri = self.add_weight(shape=param_shape,
                                            name='gamma_ri',
                                            initializer=self.gamma_off_initializer,
                                            regularizer=self.gamma_off_regularizer,
                                            constraint=self.gamma_off_constraint)
            self.moving_Vrr = self.add_weight(shape=param_shape,
                                              initializer=self.moving_variance_initializer,
                                              name='moving_Vrr',
                                              trainable=False)
            self.moving_Vii = self.add_weight(shape=param_shape,
                                              initializer=self.moving_variance_initializer,
                                              name='moving_Vii',
                                              trainable=False)
            self.moving_Vri = self.add_weight(shape=param_shape,
                                              initializer=self.moving_covariance_initializer,
                                              name='moving_Vri',
                                              trainable=False)
        else:
            self.gamma_rr = None
            self.gamma_ii = None
            self.gamma_ri = None
            self.moving_Vrr = None
            self.moving_Vii = None
            self.moving_Vri = None

        if self.center:
            self.beta = self.add_weight(shape=(input_shape[self.axis],),
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
            self.moving_mean = self.add_weight(shape=(input_shape[self.axis],),
                                               initializer=self.moving_mean_initializer,
                                               name='moving_mean',
                                               trainable=False)
        else:
            self.beta = None
            self.moving_mean = None

        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        ndim = len(input_shape)
        reduction_axes = list(range(ndim))
        del reduction_axes[self.axis]
        input_dim = input_shape[self.axis] // 2
        mu = K.mean(inputs, axis=reduction_axes)
        broadcast_mu_shape = [1] * len(input_shape)
        broadcast_mu_shape[self.axis] = input_shape[self.axis]
        broadcast_mu = K.reshape(mu, broadcast_mu_shape)
        if self.center:
            input_centred = inputs - broadcast_mu
        else:
            input_centred = inputs
        centred_squared = input_centred ** 2
        if (self.axis == 1 and ndim != 3) or ndim == 2:
            centred_squared_real = centred_squared[:, :input_dim]
            centred_squared_imag = centred_squared[:, input_dim:]
            centred_real = input_centred[:, :input_dim]
            centred_imag = input_centred[:, input_dim:]
        elif ndim == 3:
            centred_squared_real = centred_squared[:, :, :input_dim]
            centred_squared_imag = centred_squared[:, :, input_dim:]
            centred_real = input_centred[:, :, :input_dim]
            centred_imag = input_centred[:, :, input_dim:]
        elif self.axis == -1 and ndim == 4:
            centred_squared_real = centred_squared[:, :, :, :input_dim]
            centred_squared_imag = centred_squared[:, :, :, input_dim:]
            centred_real = input_centred[:, :, :, :input_dim]
            centred_imag = input_centred[:, :, :, input_dim:]
        elif self.axis == -1 and ndim == 5:
            centred_squared_real = centred_squared[:, :, :, :, :input_dim]
            centred_squared_imag = centred_squared[:, :, :, :, input_dim:]
            centred_real = input_centred[:, :, :, :, :input_dim]
            centred_imag = input_centred[:, :, :, :, input_dim:]
        else:
            raise ValueError(
                'Incorrect Batchnorm combination of axis and dimensions. axis should be either 1 or -1. '
                'axis: ' + str(self.axis) + '; ndim: ' + str(ndim) + '.'
            )
        if self.scale:
            Vrr = K.mean(
                centred_squared_real,
                axis=reduction_axes
            ) + self.epsilon
            Vii = K.mean(
                centred_squared_imag,
                axis=reduction_axes
            ) + self.epsilon
            # Vri contains the real and imaginary covariance for each feature map.
            Vri = K.mean(
                centred_real * centred_imag,
                axis=reduction_axes,
            )
        elif self.center:
            Vrr = None
            Vii = None
            Vri = None
        else:
            raise ValueError('Error. Both scale and center in batchnorm are set to False.')

        input_bn = ComplexBN(
            input_centred, Vrr, Vii, Vri,
            self.beta, self.gamma_rr, self.gamma_ri,
            self.gamma_ii, self.scale, self.center,
            axis=self.axis
        )
        if training in {0, False}:
            return input_bn
        else:
            update_list = []
            if self.center:
                update_list.append(K.moving_average_update(self.moving_mean, mu, self.momentum))
            if self.scale:
                update_list.append(K.moving_average_update(self.moving_Vrr, Vrr, self.momentum))
                update_list.append(K.moving_average_update(self.moving_Vii, Vii, self.momentum))
                update_list.append(K.moving_average_update(self.moving_Vri, Vri, self.momentum))
            self.add_update(update_list, inputs)

            def normalize_inference():
                if self.center:
                    inference_centred = inputs - K.reshape(self.moving_mean, broadcast_mu_shape)
                else:
                    inference_centred = inputs
                return ComplexBN(
                    inference_centred, self.moving_Vrr, self.moving_Vii,
                    self.moving_Vri, self.beta, self.gamma_rr, self.gamma_ri,
                    self.gamma_ii, self.scale, self.center, axis=self.axis
                )

        # Pick the normalized form corresponding to the training phase.
        return K.in_train_phase(input_bn,
                                normalize_inference,
                                training=training)

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_diag_initializer': initializers.serialize(self.gamma_diag_initializer) if self.gamma_diag_initializer != sqrt_init else 'sqrt_init',
            'gamma_off_initializer': initializers.serialize(self.gamma_off_initializer),
            'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer': initializers.serialize(self.moving_variance_initializer) if self.moving_variance_initializer != sqrt_init else 'sqrt_init',
            'moving_covariance_initializer': initializers.serialize(self.moving_covariance_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_diag_regularizer': regularizers.serialize(self.gamma_diag_regularizer),
            'gamma_off_regularizer': regularizers.serialize(self.gamma_off_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_diag_constraint': constraints.serialize(self.gamma_diag_constraint),
            'gamma_off_constraint': constraints.serialize(self.gamma_off_constraint),
        }
        base_config = super(ComplexBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

