#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Authors: Olexa Bilaniuk
#

import keras.backend                        as KB
import keras.engine                         as KE
import keras.layers                         as KL
import keras.optimizers                     as KO
import theano                               as T
import theano.ifelse                        as TI
import theano.tensor                        as TT
import theano.tensor.fft                    as TTF
import numpy                                as np


#
# FFT functions:
#
#  fft():   Batched 1-D FFT  (Input: (Batch, TimeSamples))
#  ifft():  Batched 1-D IFFT (Input: (Batch, FreqSamples))
#  fft2():  Batched 2-D FFT  (Input: (Batch, TimeSamplesH, TimeSamplesW))
#  ifft2(): Batched 2-D IFFT (Input: (Batch, FreqSamplesH, FreqSamplesW))
#

def fft(z):
	B      = z.shape[0]//2
	L      = z.shape[1]
	C      = TT.as_tensor_variable(np.asarray([[[1,-1]]], dtype=T.config.floatX))
	Zr, Zi = TTF.rfft(z[:B], norm="ortho"), TTF.rfft(z[B:], norm="ortho")
	isOdd  = TT.eq(L%2, 1)
	Zr     = TI.ifelse(isOdd, TT.concatenate([Zr, C*Zr[:,1:  ][:,::-1]], axis=1),
	                          TT.concatenate([Zr, C*Zr[:,1:-1][:,::-1]], axis=1))
	Zi     = TI.ifelse(isOdd, TT.concatenate([Zi, C*Zi[:,1:  ][:,::-1]], axis=1),
	                          TT.concatenate([Zi, C*Zi[:,1:-1][:,::-1]], axis=1))
	Zi     = (C*Zi)[:,:,::-1]  # Zi * i
	Z      = Zr+Zi
	return TT.concatenate([Z[:,:,0], Z[:,:,1]], axis=0)
def ifft(z):
	B      = z.shape[0]//2
	L      = z.shape[1]
	C      = TT.as_tensor_variable(np.asarray([[[1,-1]]], dtype=T.config.floatX))
	Zr, Zi = TTF.rfft(z[:B], norm="ortho"), TTF.rfft(z[B:]*-1, norm="ortho")
	isOdd  = TT.eq(L%2, 1)
	Zr     = TI.ifelse(isOdd, TT.concatenate([Zr, C*Zr[:,1:  ][:,::-1]], axis=1),
	                          TT.concatenate([Zr, C*Zr[:,1:-1][:,::-1]], axis=1))
	Zi     = TI.ifelse(isOdd, TT.concatenate([Zi, C*Zi[:,1:  ][:,::-1]], axis=1),
	                          TT.concatenate([Zi, C*Zi[:,1:-1][:,::-1]], axis=1))
	Zi     = (C*Zi)[:,:,::-1]  # Zi * i
	Z      = Zr+Zi
	return TT.concatenate([Z[:,:,0], Z[:,:,1]*-1], axis=0)
def fft2(x):
	tt = x
	tt = KB.reshape(tt, (x.shape[0] *x.shape[1], x.shape[2]))
	tf = fft(tt)
	tf = KB.reshape(tf, (x.shape[0], x.shape[1], x.shape[2]))
	tf = KB.permute_dimensions(tf, (0, 2, 1))
	tf = KB.reshape(tf, (x.shape[0] *x.shape[2], x.shape[1]))
	ff = fft(tf)
	ff = KB.reshape(ff, (x.shape[0], x.shape[2], x.shape[1]))
	ff = KB.permute_dimensions(ff, (0, 2, 1))
	return ff
def ifft2(x):
	ff = x
	ff = KB.permute_dimensions(ff, (0, 2, 1))
	ff = KB.reshape(ff, (x.shape[0] *x.shape[2], x.shape[1]))
	tf = ifft(ff)
	tf = KB.reshape(tf, (x.shape[0], x.shape[2], x.shape[1]))
	tf = KB.permute_dimensions(tf, (0, 2, 1))
	tf = KB.reshape(tf, (x.shape[0] *x.shape[1], x.shape[2]))
	tt = ifft(tf)
	tt = KB.reshape(tt, (x.shape[0], x.shape[1], x.shape[2]))
	return tt

#
# FFT Layers:
#
#  FFT:   Batched 1-D FFT  (Input: (Batch, FeatureMaps, TimeSamples))
#  IFFT:  Batched 1-D IFFT (Input: (Batch, FeatureMaps, FreqSamples))
#  FFT2:  Batched 2-D FFT  (Input: (Batch, FeatureMaps, TimeSamplesH, TimeSamplesW))
#  IFFT2: Batched 2-D IFFT (Input: (Batch, FeatureMaps, FreqSamplesH, FreqSamplesW))
#

class FFT(KL.Layer):
	def call(self, x, mask=None):
		a = KB.permute_dimensions(x, (1,0,2))
		a = KB.reshape(a, (x.shape[1] *x.shape[0], x.shape[2]))
		a = fft(a)
		a = KB.reshape(a, (x.shape[1], x.shape[0], x.shape[2]))
		return KB.permute_dimensions(a, (1,0,2))
class IFFT(KL.Layer):
	def call(self, x, mask=None):
		a = KB.permute_dimensions(x, (1,0,2))
		a = KB.reshape(a, (x.shape[1] *x.shape[0], x.shape[2]))
		a = ifft(a)
		a = KB.reshape(a, (x.shape[1], x.shape[0], x.shape[2]))
		return KB.permute_dimensions(a, (1,0,2))
class FFT2(KL.Layer):
	def call(self, x, mask=None):
		a = KB.permute_dimensions(x, (1,0,2,3))
		a = KB.reshape(a, (x.shape[1] *x.shape[0], x.shape[2], x.shape[3]))
		a = fft2(a)
		a = KB.reshape(a, (x.shape[1], x.shape[0], x.shape[2], x.shape[3]))
		return KB.permute_dimensions(a, (1,0,2,3))
class IFFT2(KL.Layer):
	def call(self, x, mask=None):
		a = KB.permute_dimensions(x, (1,0,2,3))
		a = KB.reshape(a, (x.shape[1] *x.shape[0], x.shape[2], x.shape[3]))
		a = ifft2(a)
		a = KB.reshape(a, (x.shape[1], x.shape[0], x.shape[2], x.shape[3]))
		return KB.permute_dimensions(a, (1,0,2,3))



#
# Tests
#
# Note: The IFFT is the conjugate of the FFT of the conjugate.
#
#     np.fft.ifft(x) == np.conj(np.fft.fft(np.conj(x)))
#

if __name__ == "__main__":
	# Numpy
	np.random.seed(1)
	L   = 19
	r   = np.random.normal(0.8, size=(L,))
	i   = np.random.normal(0.8, size=(L,))
	x   = r+i*1j
	R   = np.fft.rfft(r, norm="ortho")
	I   = np.fft.rfft(i, norm="ortho")
	X   = np.fft.fft (x, norm="ortho")
	
	if L&1:
		R   = np.concatenate([R, np.conj(R[1:  ][::-1])])
		I   = np.concatenate([I, np.conj(I[1:  ][::-1])])
	else:
		R   = np.concatenate([R, np.conj(R[1:-1][::-1])])
		I   = np.concatenate([I, np.conj(I[1:-1][::-1])])
	Y   = R+I*1j
	print np.allclose(X, Y)
	
	
	# Theano
	z   = TT.dmatrix()
	f   = T.function([z], ifft(fft(z)))
	v   = np.concatenate([np.real(x)[np.newaxis,:], np.imag(x)[np.newaxis,:]], axis=0)
	print v
	print f(v)
	print np.allclose(v, f(v))
	
	
	# Keras
	x = i = KL.Input(shape=(128, 32,32))
	x = IFFT2()(x)
	model = KE.Model([i],[x])
	
	loss  = "mse"
	opt   = KO.Adam()
	
	model.compile(opt, loss)
	model._make_train_function()
	model._make_predict_function()
	model._make_test_function()
	
	v = np.random.normal(size=(13,128,32,32))
	#print v
	V = model.predict(v)
	#print V
	print V.shape
