#!/usr/bin/env python
# -*- coding: utf-8 -*-



# Imports.
import numpy                                    as np
import os
import sys
import torch

from   torch.nn     import (Conv2d, BatchNorm2d, MaxPool2d, AdaptiveAvgPool2d,
                            AvgPool2d, ReLU, CrossEntropyLoss, Linear)

from   .functional                          import *
from   .layers                              import *
from   .layers                              import _mktuple2d



#
# Real-valued model
#

class RealResNetProj(torch.nn.Module):
	"""
	The "projection" operation that performs dimension changes in ResNet, because
	identity cannot do so.
	
	References:
	    - https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py#L31-L35
	    - https://arxiv.org/pdf/1605.07146v2.pdf
	"""
	def __init__(self, in_channels, out_channels, stride=2):
		super().__init__()
		stride    = _mktuple2d(stride)
		self.conv = Conv2d     (in_channels, out_channels, kernel_size=stride,
		                        stride=stride, padding=0, bias=False)
	
	def forward(self, x):
		return self.conv(x)

class RealResNetBB(torch.nn.Module):
	"""
	A basic B(3,3) block for ResNets.
	
	Contains two branches that are summed together:
	    1. Identity/Projection
	    2. BN-ReLU-Conv-BN-ReLU-Conv
	
	References:
	    - https://arxiv.org/pdf/1605.07146v2.pdf
	    - https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py#L25-L29
	"""
	def __init__(self, in_channels, out_channels, stride=1):
		super().__init__()
		stride = _mktuple2d(stride)
		if (in_channels != out_channels) or (stride[0] != 1) or (stride[1] != 1):
			self.proj = RealResNetProj(in_channels, out_channels, stride)
		else:
			self.proj = None
			
		self.bn1   = BatchNorm2d(in_channels, affine=True)
		self.relu1 = ReLU       ()
		self.conv1 = Conv2d     (in_channels,  out_channels, kernel_size=(3,3),
		                         stride=stride, padding=1, bias=False)
		self.bn2   = BatchNorm2d(out_channels, affine=True)
		self.relu2 = ReLU       ()
		self.conv2 = Conv2d     (out_channels, out_channels, kernel_size=(3,3),
		                         stride=1,      padding=1, bias=True)
	
	def forward(self, x):
		xresidual = self.proj(x) if self.proj else x
		x = self.conv1(self.relu1(self.bn1(x)))
		x = self.conv2(self.relu2(self.bn2(x)))
		return x+xresidual

class RealResNet34(torch.nn.Module):
	def __init__(self, a):
		super().__init__()
		self.a        = a
		
		self.conv1    = Conv2d      (3, 192, (7,7), stride=2, padding=3, bias=False)
		self.maxpool1 = MaxPool2d   (kernel_size=(3,3), stride=(2,2), padding=1)
		
		self.bb2_1    = RealResNetBB(192,  64)
		self.bb2_2    = RealResNetBB( 64,  64)
		self.bb2_3    = RealResNetBB( 64,  64)
		
		self.bb3_1    = RealResNetBB( 64, 128, stride=2)
		self.bb3_2    = RealResNetBB(128, 128)
		self.bb3_3    = RealResNetBB(128, 128)
		self.bb3_4    = RealResNetBB(128, 128)
		
		self.bb4_1    = RealResNetBB(128, 256, stride=2)
		self.bb4_2    = RealResNetBB(256, 256)
		self.bb4_3    = RealResNetBB(256, 256)
		self.bb4_4    = RealResNetBB(256, 256)
		self.bb4_5    = RealResNetBB(256, 256)
		self.bb4_6    = RealResNetBB(256, 256)
		
		self.bb5_1    = RealResNetBB(256, 512, stride=2)
		self.bb5_2    = RealResNetBB(512, 512)
		self.bb5_3    = RealResNetBB(512, 512)
		
		self.avgpool  = AdaptiveAvgPool2d(1)
		self.fc       = Linear(512, 1000)
		
		self.celoss   = torch.nn.CrossEntropyLoss()
		
		def selfinit(mod):
			if   "Conv"      in mod.__class__.__name__:
				torch.nn.init.xavier_uniform_(mod.weight, gain=np.sqrt(2))
				if mod.bias is not None:
					torch.nn.init.constant_(mod.bias,   0)
			elif "BatchNorm" in mod.__class__.__name__:
				if mod.weight is not None:
					torch.nn.init.constant_(mod.weight, 1)
				if mod.bias   is not None:
					torch.nn.init.constant_(mod.bias,   0)
		self.apply(selfinit)
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.maxpool1(x)
		x = self.bb2_3(self.bb2_2(self.bb2_1(x)))
		x = self.bb3_4(self.bb3_3(self.bb3_2(self.bb3_1(x))))
		x = self.bb4_6(self.bb4_5(self.bb4_4(self.bb4_3(self.bb4_2(self.bb4_1(x))))))
		x = self.bb5_3(self.bb5_2(self.bb5_1(x)))
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x
	
	def loss(self, Ypred, Y):
		CEL = self.celoss(Ypred, Y)
		L2P = 0.0
		
		def l2decay(mod):
			nonlocal L2P
			if   "Conv"   in mod.__class__.__name__:
				L2P = L2P+mod.weight.pow(2).reshape(-1).sum(0)
			elif "Linear" in mod.__class__.__name__:
				L2P = L2P+mod.weight.pow(2).reshape(-1).sum(0)
		
		self.apply(l2decay)
		
		OBJ = CEL + L2P*self.a.l2
		
		return OBJ, {"loss/ce":  CEL,
		             "param/l2": torch.sqrt(L2P)}


#
# Complex-valued model
#

class ComplexResNet34(torch.nn.Module):
	pass # FIXME: Fill me out!!!!

