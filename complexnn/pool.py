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
# Spectral Pooling Layer
#

class SpectralPooling1D(KL.Layer):
	def __init__(self, topf=(0,)):
		super(SpectralPooling1D, self).__init__()
		if   "topf"  in kwargs:
			self.topf  = (int  (kwargs["topf" ][0]),)
			self.topf  = (self.topf[0]//2,)
		elif "gamma" in kwargs:
			self.gamma = (float(kwargs["gamma"][0]),)
			self.gamma = (self.gamma[0]/2,)
		else:
			raise RuntimeError("Must provide either topf= or gamma= !")
	def call(self, x, mask=None):
		xshape = x._keras_shape
		if hasattr(self, "topf"):
			topf = self.topf
		else:
			if KB.image_data_format() == "channels_first":
				topf = (int(self.gamma[0]*xshape[2]),)
			else:
				topf = (int(self.gamma[0]*xshape[1]),)
		
		if KB.image_data_format() == "channels_first":
			if topf[0] > 0 and xshape[2] >= 2*topf[0]:
				mask = [1]*(topf[0]              ) +\
					   [0]*(xshape[2] - 2*topf[0]) +\
					   [1]*(topf[0]              )
				mask = [[mask]]
				mask = np.asarray(mask, dtype=KB.floatx()).transpose((0,1,2))
				mask = KB.constant(mask)
				x   *= mask
		else:
			if topf[0] > 0 and xshape[1] >= 2*topf[0]:
				mask = [1]*(topf[0]              ) +\
					   [0]*(xshape[1] - 2*topf[0]) +\
					   [1]*(topf[0]              )
				mask = [[mask]]
				mask = np.asarray(mask, dtype=KB.floatx()).transpose((0,2,1))
				mask = KB.constant(mask)
				x   *= mask
		
		return x
class SpectralPooling2D(KL.Layer):
	def __init__(self, **kwargs):
		super(SpectralPooling2D, self).__init__()
		if   "topf"  in kwargs:
			self.topf  = (int  (kwargs["topf" ][0]), int  (kwargs["topf" ][1]))
			self.topf  = (self.topf[0]//2, self.topf[1]//2)
		elif "gamma" in kwargs:
			self.gamma = (float(kwargs["gamma"][0]), float(kwargs["gamma"][1]))
			self.gamma = (self.gamma[0]/2, self.gamma[1]/2)
		else:
			raise RuntimeError("Must provide either topf= or gamma= !")
	def call(self, x, mask=None):
		xshape = x._keras_shape
		if hasattr(self, "topf"):
			topf = self.topf
		else:
			if KB.image_data_format() == "channels_first":
				topf = (int(self.gamma[0]*xshape[2]), int(self.gamma[1]*xshape[3]))
			else:
				topf = (int(self.gamma[0]*xshape[1]), int(self.gamma[1]*xshape[2]))
		
		if KB.image_data_format() == "channels_first":
			if topf[0] > 0 and xshape[2] >= 2*topf[0]:
				mask = [1]*(topf[0]              ) +\
					   [0]*(xshape[2] - 2*topf[0]) +\
					   [1]*(topf[0]              )
				mask = [[[mask]]]
				mask = np.asarray(mask, dtype=KB.floatx()).transpose((0,1,3,2))
				mask = KB.constant(mask)
				x   *= mask
			if topf[1] > 0 and xshape[3] >= 2*topf[1]:
				mask = [1]*(topf[1]              ) +\
					   [0]*(xshape[3] - 2*topf[1]) +\
					   [1]*(topf[1]              )
				mask = [[[mask]]]
				mask = np.asarray(mask, dtype=KB.floatx()).transpose((0,1,2,3))
				mask = KB.constant(mask)
				x   *= mask
		else:
			if topf[0] > 0 and xshape[1] >= 2*topf[0]:
				mask = [1]*(topf[0]              ) +\
					   [0]*(xshape[1] - 2*topf[0]) +\
					   [1]*(topf[0]              )
				mask = [[[mask]]]
				mask = np.asarray(mask, dtype=KB.floatx()).transpose((0,3,1,2))
				mask = KB.constant(mask)
				x   *= mask
			if topf[1] > 0 and xshape[2] >= 2*topf[1]:
				mask = [1]*(topf[1]              ) +\
					   [0]*(xshape[2] - 2*topf[1]) +\
					   [1]*(topf[1]              )
				mask = [[[mask]]]
				mask = np.asarray(mask, dtype=KB.floatx()).transpose((0,1,3,2))
				mask = KB.constant(mask)
				x   *= mask
		
		return x


if __name__ == "__main__":
	import cv2, sys
	import __main__ as SP
	import fft      as CF
	
	# Build Model
	x = i = KL.Input(shape=(6,512,512))
	f = CF.FFT2()(x)
	p = SP.SpectralPooling2D(gamma=[0.15,0.15])(f)
	o = CF.IFFT2()(p)
	
	model = KE.Model([i], [f,p,o])
	model.compile("sgd", "mse")
	
	# Use it
	img      = cv2.imread(sys.argv[1])
	imgBatch = img[np.newaxis,...].transpose((0,3,1,2))
	imgBatch = np.concatenate([imgBatch, np.zeros_like(imgBatch)], axis=1)
	f,p,o    = model.predict(imgBatch)
	ffted    = np.sqrt(np.sum(f[:,:3]**2 + f[:,3:]**2, axis=1))
	ffted    = ffted .transpose((1,2,0))/255
	pooled   = np.sqrt(np.sum(p[:,:3]**2 + p[:,3:]**2, axis=1))
	pooled   = pooled.transpose((1,2,0))/255
	filtered = np.clip(o,0,255).transpose((0,2,3,1))[0,:,:,:3].astype("uint8")
	
	# Display it
	cv2.imshow("Original", img)
	cv2.imshow("FFT",      ffted)
	cv2.imshow("Pooled",   pooled)
	cv2.imshow("Filtered", filtered)
	cv2.waitKey(0)
