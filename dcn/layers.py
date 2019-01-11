# -*- coding: utf-8 -*-
import numpy           as np
import scipy.linalg
import torch

from   .functional import *



#
# Utility functions
#
def _istuple(x):   return isinstance(x, tuple)
def _mktuple2d(x): return x if _istuple(x) else (x,x)

#
# Layers
#

class ComplexBatchNorm(torch.nn.Module):
	"""Mostly copied/inspired from PyTorch torch/nn/modules/batchnorm.py"""
	def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
	             track_running_stats=True):
		super(ComplexBatchNorm, self).__init__()
		self.num_features        = num_features
		self.eps                 = eps
		self.momentum            = momentum
		self.affine              = affine
		self.track_running_stats = track_running_stats
		if self.affine:
			self.Wrr = torch.nn.Parameter(torch.Tensor(num_features))
			self.Wri = torch.nn.Parameter(torch.Tensor(num_features))
			self.Wii = torch.nn.Parameter(torch.Tensor(num_features))
			self.Br  = torch.nn.Parameter(torch.Tensor(num_features))
			self.Bi  = torch.nn.Parameter(torch.Tensor(num_features))
		else:
			self.register_parameter('Wrr', None)
			self.register_parameter('Wri', None)
			self.register_parameter('Wii', None)
			self.register_parameter('Br',  None)
			self.register_parameter('Bi',  None)
		if self.track_running_stats:
			self.register_buffer('RMr',  torch.zeros(num_features))
			self.register_buffer('RMi',  torch.zeros(num_features))
			self.register_buffer('RVrr', torch.ones (num_features))
			self.register_buffer('RVri', torch.zeros(num_features))
			self.register_buffer('RVii', torch.ones (num_features))
			self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
		else:
			self.register_parameter('RMr',                 None)
			self.register_parameter('RMi',                 None)
			self.register_parameter('RVrr',                None)
			self.register_parameter('RVri',                None)
			self.register_parameter('RVii',                None)
			self.register_parameter('num_batches_tracked', None)
		self.reset_parameters()
	
	def reset_running_stats(self):
		if self.track_running_stats:
			self.RMr .zero_()
			self.RMi .zero_()
			self.RVrr.fill_(1)
			self.RVri.zero_()
			self.RVii.fill_(1)
			self.num_batches_tracked.zero_()
	
	def reset_parameters(self):
		self.reset_running_stats()
		if self.affine:
			self.Br .data.zero_()
			self.Bi .data.zero_()
			self.Wrr.data.fill_(1)
			self.Wri.data.uniform_(-.9, +.9) # W will be positive-definite
			self.Wii.data.fill_(1)
	
	def _check_input_dim(self, xr, xi):
		assert(xr.shape == xi.shape)
		assert(xr.size(1) == self.num_features)
	
	def forward(self, xr, xi):
		self._check_input_dim(xr, xi)
		
		exponential_average_factor = 0.0
		
		if self.training and self.track_running_stats:
			self.num_batches_tracked += 1
			if self.momentum is None:  # use cumulative moving average
				exponential_average_factor = 1.0 / self.num_batches_tracked.item()
			else:  # use exponential moving average
				exponential_average_factor = self.momentum
		
		#
		# NOTE: The precise meaning of the "training flag" is:
		#       True:  Normalize using batch   statistics, update running statistics
		#              if they are being collected.
		#       False: Normalize using running statistics, ignore batch   statistics.
		#
		training = self.training or not self.track_running_stats
		redux = [i for i in reversed(range(xr.dim())) if i!=1]
		vdim  = [1]*xr.dim()
		vdim[1] = xr.size(1)
		
		#
		# Mean M Computation and Centering
		#
		# Includes running mean update if training and running.
		#
		if training:
			Mr = xr
			Mi = xi
			for d in redux:
				Mr = Mr.mean(d, keepdim=True)
				Mi = Mi.mean(d, keepdim=True)
			if self.track_running_stats:
				self.RMr.lerp_(Mr.squeeze(), exponential_average_factor)
				self.RMi.lerp_(Mi.squeeze(), exponential_average_factor)
		else:
			Mr = self.RMr.view(vdim)
			Mi = self.RMi.view(vdim)
		xr, xi = xr-Mr, xi-Mi
		
		#
		# Variance Matrix V Computation
		# 
		# Includes epsilon numerical stabilizer/Tikhonov regularizer.
		# Includes running variance update if training and running.
		#
		if training:
			Vrr = xr*xr
			Vri = xr*xi
			Vii = xi*xi
			for d in redux:
				Vrr = Vrr.mean(d, keepdim=True)
				Vri = Vri.mean(d, keepdim=True)
				Vii = Vii.mean(d, keepdim=True)
			if self.track_running_stats:
				self.RVrr.lerp_(Vrr.squeeze(), exponential_average_factor)
				self.RVri.lerp_(Vri.squeeze(), exponential_average_factor)
				self.RVii.lerp_(Vii.squeeze(), exponential_average_factor)
		else:
			Vrr = self.RVrr.view(vdim)
			Vri = self.RVri.view(vdim)
			Vii = self.RVii.view(vdim)
		Vrr   = Vrr+self.eps
		Vri   = Vri
		Vii   = Vii+self.eps
		
		#
		# Matrix Inverse Square Root U = V^-0.5
		#
		tau   = Vrr+Vii
		delta = torch.addcmul(Vrr*Vii, -1, Vri, Vri)
		s     = delta.sqrt()
		t     = (tau + 2*s).sqrt()
		rst   = (s*t).reciprocal()
		
		Urr   = (s+Vii)*rst
		Uii   = (s+Vrr)*rst
		Uri   = ( -Vri)*rst
		
		#
		# Optionally left-multiply U by affine weights W to produce combined
		# weights Z, left-multiply the inputs by Z, then optionally bias them.
		#
		# y = Zx + B
		# y = WUx + B
		# y = [Wrr Wri][Urr Uri] [xr] + [Br]
		#     [Wir Wii][Uir Uii] [xi]   [Bi]
		#
		if self.affine:
			Zrr = self.Wrr*Urr + self.Wri*Uri
			Zri = self.Wrr*Uri + self.Wri*Uii
			Zir = self.Wri*Urr + self.Wii*Uri
			Zii = self.Wri*Uri + self.Wii*Uii
		else:
			Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii
			
		yr, yi = Zrr*xr + Zri*xi, Zir*xr + Zii*xi
		
		if self.affine:
			yr = yr + self.Br
			yi = yi + self.Bi
		
		return yr, yi
	
	def extra_repr(self):
		return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
			   'track_running_stats={track_running_stats}'.format(**self.__dict__)
	
	def _load_from_state_dict(self, state_dict, prefix, strict, missing_keys,
	                                unexpected_keys, error_msgs):
		super(ComplexBatchNorm, self)._load_from_state_dict(state_dict,
		                                                    prefix,
		                                                    strict,
		                                                    missing_keys,
		                                                    unexpected_keys,
		                                                    error_msgs)


class ComplexConv2d(torch.nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
	             padding=0, dilation=1, groups=1, bias=True):
		super(ComplexConv2d, self).__init__()
		self.in_channels  = in_channels
		self.out_channels = out_channels
		self.kernel_size  = _mktuple2d(kernel_size)
		self.stride       = _mktuple2d(stride)
		self.padding      = _mktuple2d(padding)
		self.dilation     = _mktuple2d(dilation)
		self.groups       = groups
		
		self.Wr           = torch.nn.Parameter(torch.Tensor(self.out_channels,
		                                                    self.in_channels // self.groups,
		                                                    *self.kernel_size))
		self.Wi           = torch.nn.Parameter(torch.Tensor(self.out_channels,
		                                                    self.in_channels // self.groups,
		                                                    *self.kernel_size))
		if bias:
			self.Br = torch.nn.Parameter(torch.Tensor(out_channels))
			self.Bi = torch.nn.Parameter(torch.Tensor(out_channels))
		else:
			self.register_parameter("Br", None)
			self.register_parameter("Bi", None)
		self.reset_parameters()
	
	def reset_parameters(self):
		fanin = self.in_channels // self.groups
		for s in self.kernel_size: fanin *= s
		complex_rayleigh_init(self.Wr, self.Wi, fanin)
		if self.Br is not None and self.Bi is not None:
			self.Br.data.zero_()
			self.Bi.data.zero_()
	
	def forward(self, xr, xi):
		yrr = torch.nn.functional.conv2d(xr, self.Wr,  self.Br,       self.stride,
		                                 self.padding, self.dilation, self.groups)
		yri = torch.nn.functional.conv2d(xr, self.Wi,  self.Bi,       self.stride,
		                                 self.padding, self.dilation, self.groups)
		yir = torch.nn.functional.conv2d(xi, self.Wr,  None,          self.stride,
		                                 self.padding, self.dilation, self.groups)
		yii = torch.nn.functional.conv2d(xi, self.Wi,  None,          self.stride,
		                                 self.padding, self.dilation, self.groups)
		return yrr-yii, yri+yir


class ComplexLinear(torch.nn.Module):
	def __init__(self, in_features, out_features, bias=True):
		super(ComplexLinear, self).__init__()
		self.in_features  = in_features
		self.out_features = out_features
		self.Wr           = torch.nn.Parameter(torch.Tensor(out_features, in_features))
		self.Wi           = torch.nn.Parameter(torch.Tensor(out_features, in_features))
		if bias:
			self.Br = torch.nn.Parameter(torch.Tensor(out_features))
			self.Bi = torch.nn.Parameter(torch.Tensor(out_features))
		else:
			self.register_parameter('Br', None)
			self.register_parameter('Bi', None)
		self.reset_parameters()
	
	def reset_parameters(self):
		complex_rayleigh_init(self.Wr, self.Wi, self.in_features)
		if self.Br is not None and self.Bi is not None:
			self.Br.data.zero_()
			self.Bi.data.zero_()
	
	def forward(self, xr, xi):
		yrr = torch.nn.functional.linear(xr, self.Wr, self.Br)
		yri = torch.nn.functional.linear(xr, self.Wi, self.Bi)
		yir = torch.nn.functional.linear(xi, self.Wr, None)
		yii = torch.nn.functional.linear(xi, self.Wi, None)
		return yrr-yii, yri+yir


class STFT2d(torch.nn.Module):
	def __init__(self, in_channels, kernel_size=8, window_fn=np.hamming, stride=1,
	             padding=0, dilation=1, inverse=False):
		super(STFT2d, self).__init__()
		self.in_channels  = in_channels
		self.kernel_size  = _mktuple2d(kernel_size)
		self.window_fn    = _mktuple2d(window_fn)
		self.stride       = _mktuple2d(stride)
		self.padding      = _mktuple2d(padding)
		self.dilation     = _mktuple2d(dilation)
		self.inverse      = bool(inverse);
		
		h  = self.kernel_size[0]
		w  = self.kernel_size[1]
		wH = self.window_fn[0](h)
		wW = self.window_fn[1](w)
		Fh = scipy.linalg.dft(h, "sqrtn").astype("complex128")
		Fw = scipy.linalg.dft(w, "sqrtn").astype("complex128")
		
		F  = np.einsum("i,j,fi,gj->fgij", wH, wW, Fh, Fw)
		F  = F.astype("complex128")
		F  = F.reshape(-1,1,h,w)
		F  = F.conjugate() if self.inverse else F
		
		Wr = torch.empty(*F.real.shape).copy_(torch.from_numpy(F.real))
		Wi = torch.empty(*F.imag.shape).copy_(torch.from_numpy(F.imag))
		self.register_buffer("Wr", Wr)
		self.register_buffer("Wi", Wi)
	
	def forward(self, xr, xi=None):
		inpSize = xr.shape
		B  = inpSize[0]
		if self.inverse:
			assert(xi is not None)
			xr = xr.view(B*self.in_channels, -1, *inpSize[2:])
			xi = xi.view(B*self.in_channels, -1, *inpSize[2:])
			rr = torch.nn.functional.conv_transpose2d(xr,
			                                          self.Wr,
			                                          None,
			                                          stride  =self.stride,
			                                          padding =self.padding,
			                                          dilation=self.dilation)
			ri = torch.nn.functional.conv_transpose2d(xr,
			                                          self.Wi,
			                                          None,
			                                          stride  =self.stride,
			                                          padding =self.padding,
			                                          dilation=self.dilation)
			ir = torch.nn.functional.conv_transpose2d(xi,
			                                          self.Wr,
			                                          None,
			                                          stride  =self.stride,
			                                          padding =self.padding,
			                                          dilation=self.dilation)
			ii = torch.nn.functional.conv_transpose2d(xi,
			                                          self.Wi,
			                                          None,
			                                          stride  =self.stride,
			                                          padding =self.padding,
			                                          dilation=self.dilation)
			rr = rr.view(B, -1, *rr.shape[2:])
			ri = ri.view(B, -1, *ri.shape[2:])
			ir = ir.view(B, -1, *ir.shape[2:])
			ii = ii.view(B, -1, *ii.shape[2:])
			return rr-ii, ri+ir
		else:
			xr = xr.view(B*self.in_channels, 1, *inpSize[2:])
			rr = torch.nn.functional.conv2d(xr,
			                                self.Wr,
			                                None,
			                                stride  =self.stride,
			                                padding =self.padding,
			                                dilation=self.dilation)
			ri = torch.nn.functional.conv2d(xr,
			                                self.Wi,
			                                None,
			                                stride  =self.stride,
			                                padding =self.padding,
			                                dilation=self.dilation)
			rr = rr.view(B, -1, *rr.shape[2:])
			ri = ri.view(B, -1, *ri.shape[2:])
			
			if xi is None:
				return rr, ri
			else:
				xi = xi.view(B*self.in_channels, 1, *inpSize[2:])
				ir = torch.nn.functional.conv2d(xi,
				                                self.Wr,
				                                None,
				                                stride  =self.stride,
				                                padding =self.padding,
				                                dilation=self.dilation)
				ii = torch.nn.functional.conv2d(xi,
				                                self.Wi,
				                                None,
				                                stride  =self.stride,
				                                padding =self.padding,
				                                dilation=self.dilation)
				ir = ir.view(B, -1, *ir.shape[2:])
				ii = ii.view(B, -1, *ii.shape[2:])
				return rr-ii, ri+ir

