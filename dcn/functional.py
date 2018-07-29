# -*- coding: utf-8 -*-
import math
import numpy as np
import torch


#
# Utility functions for initialization
#
def complex_rayleigh_init(Wr, Wi, fanin=None, gain=1):
	if not fanin:
		fanin = 1
		for p in W1.shape[1:]: fanin *= p
	scale = float(gain)/float(fanin)
	theta  = torch.empty_like(Wr).uniform_(-math.pi/2, +math.pi/2)
	rho    = np.random.rayleigh(scale, tuple(Wr.shape))
	rho    = torch.tensor(rho).to(Wr)
	Wr.data.copy_(rho*theta.cos())
	Wi.data.copy_(rho*theta.sin())

