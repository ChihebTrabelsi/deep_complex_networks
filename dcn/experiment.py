# -*- coding: utf-8 -*-
import benzina.torch
import nauka
import os, sys, time
import torch
import torchvision
import uuid

from   benzina.torch                    import (ImageNet, NvdecodeDataLoader)
from   torch.nn                         import (DataParallel,)
from   torch.nn.parallel                import (data_parallel,)
from   torch.optim                      import (SGD, RMSprop, Adam,)
from   torch.utils.data                 import (DataLoader,)
from   torch.utils.data.sampler         import (SubsetRandomSampler,)
from   torchvision.datasets             import (MNIST, CIFAR10, CIFAR100, SVHN,)
from   torchvision.transforms           import (Compose, ToTensor,)
from   zvit                             import *

#
# Local Imports:
#
from   .models                          import (RealResNet34, ComplexResNet34,)




class Experiment(nauka.exp.Experiment):
	def __init__(self, a):
		self.a = type(a)(**a.__dict__)
		self.a.__dict__.pop("__argp__", None)
		self.a.__dict__.pop("__argv__", None)
		self.a.__dict__.pop("__cls__",  None)
		if self.a.workDir:
			super().__init__(self.a.workDir)
		else:
			projName = "DeepComplexNetworks-350ba773-6abf-44f1-b0c6-0d6101aea629"
			workDir  = nauka.fhs.createWorkDir(self.a.baseDir,
			                                   projName,
			                                   self.uuid,
			                                   self.a.name)
			super().__init__(workDir)
		self.mkdirp(self.logDir)
	
	def fromScratch(self):
		"""
		Reinitialize from scratch.
		"""
		pass
		
		"""Reseed PRNGs for initialization step"""
		self.reseed(password="Seed: {} Init".format(self.a.seed))
		
		
		"""Create snapshottable-state object"""
		self.S = nauka.utils.PlainObject()
		
		
		"""Model Instantiation"""
		self.S.model = None
		if   self.a.model == "real":         self.S.model = RealResNet34   (self.a)
		elif self.a.model == "complex":      self.S.model = ComplexResNet34(self.a)
		if   self.S.model is None:
			raise ValueError("Unsupported dataset-model pair \""+self.a.dataset+"-"+self.a.model+"\"!")
		
		if self.a.cuda: self.S.model = self.S.model.cuda(self.a.cuda[0])
		else:           self.S.model = self.S.model.cpu()
		
		
		"""Optimizer Selection"""
		self.S.lr        = nauka.utils.lr.fromSpecList     ([self.a.optimizer.lr]+self.a.lr)
		self.S.optimizer = nauka.utils.torch.optim.fromSpec(self.S.model.parameters(),
		                                                    self.a.optimizer)
		nauka.utils.torch.optim.setLR(self.S.optimizer, self.S.lr)
		
		
		"""Counters"""
		self.S.epochNum    = 0
		self.S.intervalNum = 0
		self.S.zvitStepNum = 0
		
		
		return self
	
	def run(self):
		"""
		Run by intervals until experiment completion.
		"""
		with ZvitWriter(self.logDir, self.S.zvitStepNum) as self.z:
			self.readyDataset(download=False)
			while not self.isDone:
				self.interval().snapshot().purge()
		return self
	
	def interval(self):
		"""
		An interval is defined as the computation- and time-span between two
		snapshots.
		
		Hard requirements:
		- By definition, one may not invoke snapshot() within an interval.
		- Corollary: The work done by an interval is either fully recorded or
		  not recorded at all.
		- There must be a step of the event logger between any TensorBoard
		  summary log and the end of the interval.
		
		For reproducibility purposes, all PRNGs are reseeded at the beginning
		of every interval.
		"""
		
		self.onIntervalBegin()
		
		with tagscope("train"):
			self.S.model.train()
			self.onTrainLoopBegin()
			for i, D in enumerate(self.DloaderTrain):
				if self.a.fastdebug and i>=self.a.fastdebug: break
				if i>0: self.z.step()
				self.onTrainBatch(D, i)
			self.onTrainLoopEnd()
		
		with tagscope("valid"):
			self.S.model.eval()
			self.onValidLoopBegin()
			for i, D in enumerate(self.DloaderValid):
				if self.a.fastdebug and i>=self.a.fastdebug: break
				self.onValidBatch(D, i)
			self.onValidLoopEnd()
		
		self.onIntervalEnd()
		return self
	
	def onTrainBatch(self, D, i):
		X, Y = D
		
		self.S.optimizer.zero_grad()
		if self.a.cuda:
			Y, X  = Y.cuda(self.a.cuda[0]), X.cuda(self.a.cuda[0])
			X.requires_grad_()
			Ypred = data_parallel(self.S.model, X, self.a.cuda)
		else:
			Y, X  = Y.cpu(),  X.cpu()
			X.requires_grad_()
			Ypred = self.S.model(X)
		loss = self.S.model.loss(Ypred, Y)
		loss[0].backward()
		gradInput        = X.grad.pow(2).sum(3).sum(2).sum(1).sqrt().mean(0)
		gradMagUnclipped = self.getGradMagnitude()
		if self.a.clipval:
			torch.nn.utils.clip_grad_value_(self.S.model.parameters(), self.a.clipval)
		if self.a.clipnorm:
			torch.nn.utils.clip_grad_norm_ (self.S.model.parameters(), self.a.clipnorm)
		gradMagClipped   = self.getGradMagnitude()
		self.S.optimizer.step()
		
		with torch.no_grad():
			with tagscope("batch"):
				with tagscope("grad"):
					with tagscope("input"):
						logScalar("l2",           gradInput)
					with tagscope("param"):
						logScalar("unclipped/l2", gradMagUnclipped)
						logScalar("master/l2",    gradMagClipped)
				self.recordTrainBatchStats(X, Ypred, Y, loss)
		
		return self
	
	def onValidBatch(self, D, i):
		X, Y = D
		
		with torch.no_grad():
			if self.a.cuda:
				Y, X  = Y.cuda(self.a.cuda[0]), X.cuda(self.a.cuda[0])
				Ypred = data_parallel(self.S.model, X, self.a.cuda)
			else:
				Y, X  = Y.cpu(),  X.cpu()
				Ypred = self.S.model(X)
			loss = self.S.model.loss(Ypred, Y)
		
		with torch.no_grad():
			self.recordValidBatchStats(X, Ypred, Y, loss)
		
		return self
	
	def recordTrainBatchStats(self, X, Ypred, Y, loss):
		batchSize = Y.size(0)
		
		self.S.totalTrainLoss += float(loss[0]*batchSize)
		self.S.totalTrainErr  += int(torch.max(Ypred, 1)[1].ne(Y).long().sum())
		self.S.totalTrainCnt  += batchSize
		batchEndTime           = time.time()
		batchTime              = batchEndTime-self.batchStartTime
		self.batchStartTime    = batchEndTime
		logScalar("loss/master", loss[0])
		logScalar("time",  batchTime)
		logScalars(loss[1])
	
	def recordValidBatchStats(self, X, Ypred, Y, loss):
		batchSize = Y.size(0)
		self.S.totalValidLoss += float(loss[1]["loss/ce"]*batchSize)
		self.S.totalValidErr  += int(torch.max(Ypred, 1)[1].ne(Y).long().sum())
		self.S.totalValidCnt  += batchSize
	
	def onTrainLoopBegin(self):
		self.S.totalTrainLoss = 0
		self.S.totalTrainErr  = 0
		self.S.totalTrainCnt  = 0
		self.epochStartTime   = self.batchStartTime = time.time()
		return self
	
	def onTrainLoopEnd(self):
		with tagscope("epoch"):
			logScalar("loss", self.S.totalTrainLoss/self.S.totalTrainCnt)
			logScalar("err",  self.S.totalTrainErr /self.S.totalTrainCnt)
		return self
	
	def onValidLoopBegin(self):
		self.S.totalValidLoss = 0
		self.S.totalValidErr  = 0
		self.S.totalValidCnt  = 0
		return self
	
	def onValidLoopEnd(self):
		with tagscope("epoch"):
			valLoss = self.S.totalValidLoss/self.S.totalValidCnt
			valErr  = self.S.totalValidErr /self.S.totalValidCnt
			logScalar("loss", valLoss)
			logScalar("err",  valErr)
		self.S.lr.step(metric=valErr)
		return self
	
	def onIntervalBegin(self):
		self.reseed()
		self.readyLoaders()
		nauka.utils.torch.optim.setLR(self.S.optimizer, self.S.lr)
		logScalar("lr", self.S.lr)
		return self
	
	def onIntervalEnd(self):
		sys.stdout.write("Epoch {:d} done.\n".format(self.S.epochNum))
		self.S.epochNum    += 1
		self.S.intervalNum += 1
		self.z.step()
		return self
	
	def getGradMagnitude(self, p=2):
		with torch.no_grad():
			M = 0.0
			for param in self.S.model.parameters():
				M = M+param.grad.abs().pow(p).reshape(-1).sum(0)
			return torch.pow(M, 1.0/p)
	
	def reseed(self, password=None):
		"""
		Reseed PRNGs for reproducibility at beginning of interval.
		"""
		#
		# The "password" from which the seeds are derived should be unique per
		# interval to ensure different seedings. Given the same
		#   - password
		#   - salt
		#   - rounds #
		#   - hash function
		# , the reproduced seed will always be the same.
		#
		# We choose as salt the PRNG's name. Since it's different for every
		# PRNG, their sequences will be different, even if they share the same
		# "password".
		#
		password = password or "Seed: {} Interval: {:d}".format(self.a.seed,
		                                                        self.S.intervalNum,)
		nauka.utils.random.setstate           (password)
		nauka.utils.numpy.random.set_state    (password)
		nauka.utils.torch.random.manual_seed  (password)
		nauka.utils.torch.cuda.manual_seed_all(password)
		return self
	
	def readyDataset(self, download=False):
		"""
		Ready the datasets, downloading or copying if necessary, permitted and
		able.
		"""
		if   self.a.dataset == "mnist":
			self.Dxform    = [ToTensor()]
			self.Dxform    = Compose(self.Dxform)
			self.DsetTrain = MNIST   (self.dataDir, True,    self.Dxform, download=download)
			self.DsetTest  = MNIST   (self.dataDir, False,   self.Dxform, download=download)
			self.DsetValid = self.DsetTrain
			self.Dimgsz    = (1, 28, 28)
			self.DNclass   = 10
			self.DNtrnvld  = len(self.DsetTrain)
			self.DNvalid   = 5000
			self.DNtest    = len(self.DsetTest)
			self.DNtrain   = self.DNtrnvld-self.DNvalid
			self.DidxTrain = range(self.DNtrnvld)[:self.DNtrain]
			self.DidxValid = range(self.DNtrnvld)[-self.DNvalid:]
			self.DidxTest  = range(self.DNtest)
		elif self.a.dataset == "cifar10":
			self.Dxform    = [ToTensor()]
			self.Dxform    = Compose(self.Dxform)
			self.DsetTrain = CIFAR10 (self.dataDir, True,    self.Dxform, download=download)
			self.DsetTest  = CIFAR10 (self.dataDir, False,   self.Dxform, download=download)
			self.DsetValid = self.DsetTrain
			self.Dimgsz    = (3, 32, 32)
			self.DNclass   = 10
			self.DNtrnvld  = len(self.DsetTrain)
			self.DNvalid   = 5000
			self.DNtest    = len(self.DsetTest)
			self.DNtrain   = self.DNtrnvld-self.DNvalid
			self.DidxTrain = range(self.DNtrnvld)[:self.DNtrain]
			self.DidxValid = range(self.DNtrnvld)[-self.DNvalid:]
			self.DidxTest  = range(self.DNtest)
		elif self.a.dataset == "cifar100":
			self.Dxform    = [ToTensor()]
			self.Dxform    = Compose(self.Dxform)
			self.DsetTrain = CIFAR100(self.dataDir, True,    self.Dxform, download=download)
			self.DsetTest  = CIFAR100(self.dataDir, False,   self.Dxform, download=download)
			self.DsetValid = self.DsetTrain
			self.Dimgsz    = (3, 32, 32)
			self.DNclass   = 100
			self.DNtrnvld  = len(self.DsetTrain)
			self.DNvalid   = 5000
			self.DNtest    = len(self.DsetTest)
			self.DNtrain   = self.DNtrnvld-self.DNvalid
			self.DidxTrain = range(self.DNtrnvld)[:self.DNtrain]
			self.DidxValid = range(self.DNtrnvld)[-self.DNvalid:]
			self.DidxTest  = range(self.DNtest)
		elif self.a.dataset == "svhn":
			self.Dxform    = [ToTensor()]
			self.Dxform    = Compose(self.Dxform)
			self.DsetTrain = SVHN    (self.dataDir, "train", self.Dxform, download=download)
			self.DsetTest  = SVHN    (self.dataDir, "test",  self.Dxform, download=download)
			self.DsetValid = self.DsetTrain
			self.Dimgsz    = (3, 32, 32)
			self.DNclass   = 10
			self.DNtrnvld  = len(self.DsetTrain)
			self.DNvalid   = 5000
			self.DNtest    = len(self.DsetTest)
			self.DNtrain   = self.DNtrnvld-self.DNvalid
			self.DidxTrain = range(self.DNtrnvld)[:self.DNtrain]
			self.DidxValid = range(self.DNtrnvld)[-self.DNvalid:]
			self.DidxTest  = range(self.DNtest)
		elif self.a.dataset == "imagenet":
			self.DsetTrain = ImageNet(self.dataDir)
			self.DsetValid = self.DsetTrain
			self.DsetTest  = self.DsetTrain
			self.Dimgsz    = (3, 224, 224)
			self.DNclass   = 10
			self.DNvalid   = 50000
			self.DNtest    = 50000
			self.DNtrain   = len(self.DsetTrain)-100000-self.DNvalid-self.DNtest
			self.DidxTrain = range(self.DNtrain)
			self.DidxValid = range(self.DNtrain,
			                       self.DNtrain+self.DNvalid)
			self.DidxTest  = range(self.DNtrain+self.DNvalid,
			                       self.DNtrain+self.DNvalid+self.DNtest)
		else:
			raise ValueError("Unknown dataset \""+self.a.dataset+"\"!")
		return self
	
	def readyLoaders(self):
		"""
		Ready the data loaders reproducibly, knowing and relying on the fact
		that PRNG states have been reproduced.
		"""
		self.DsamplerTrain = SubsetRandomSampler(self.DidxTrain)
		self.DsamplerValid = SubsetRandomSampler(self.DidxValid)
		self.DsamplerTest  = SubsetRandomSampler(self.DidxTest)
		if self.a.dataset == "imagenet":
			imagenetWarpTrain = benzina.torch.nvdecode.NvdecodeSimilarityTransform(
			    tx=( 0,32), ty=( 0,32), reflecth=0.5, autoscale=False
			)
			imagenetWarpValid = benzina.torch.nvdecode.NvdecodeSimilarityTransform(
			    tx=(16,16), ty=(16,16), reflecth=0.0, autoscale=False
			)
			imagenetWarpTest  = benzina.torch.nvdecode.NvdecodeSimilarityTransform(
			    tx=(16,16), ty=(16,16), reflecth=0.0, autoscale=False
			)
			imagenetScale   = (5094.0579569570145**-.5, 4840.219848480945**-.5, 5285.324770813224**-.5)
			imagenetBias    = ( 123.46163626781416,      116.68808194848208,     103.80484430987059)
			self.DloaderTrain  = NvdecodeDataLoader(dataset     = self.DsetTrain,
			                                        batch_size  = self.a.batch_size,
			                                        shuffle     = False,
			                                        sampler     = self.DsamplerTrain,
			                                        shape       = (224, 224),
			                                        device_id   = self.a.cuda[0],
			                                        warp_transform  = imagenetWarpTrain,
			                                        color_transform = 1,
			                                        scale_transform = imagenetScale,
			                                        bias_transform  = imagenetBias)
			self.DloaderValid  = NvdecodeDataLoader(dataset     = self.DsetValid,
			                                        batch_size  = self.a.batch_size,
			                                        shuffle     = False,
			                                        sampler     = self.DsamplerValid,
			                                        shape       = (224, 224),
			                                        device_id   = self.a.cuda[0],
			                                        warp_transform  = imagenetWarpValid,
			                                        color_transform = 1,
			                                        scale_transform = imagenetScale,
			                                        bias_transform  = imagenetBias)
			self.DloaderTest   = NvdecodeDataLoader(dataset     = self.DsetTest,
			                                        batch_size  = self.a.batch_size,
			                                        shuffle     = False,
			                                        sampler     = self.DsamplerTest,
			                                        shape       = (224, 224),
			                                        device_id   = self.a.cuda[0],
			                                        warp_transform  = imagenetWarpTest,
			                                        color_transform = 1,
			                                        scale_transform = imagenetScale,
			                                        bias_transform  = imagenetBias)
		else:
			self.DloaderTrain  = DataLoader(dataset     = self.DsetTrain,
			                                batch_size  = self.a.batch_size,
			                                shuffle     = False,
			                                sampler     = self.DsamplerTrain,
			                                num_workers = 0,
			                                pin_memory  = False)
			self.DloaderValid  = DataLoader(dataset     = self.DsetValid,
			                                batch_size  = self.a.batch_size,
			                                shuffle     = False,
			                                sampler     = self.DsamplerValid,
			                                num_workers = 0,
			                                pin_memory  = False)
			self.DloaderTest   = DataLoader(dataset     = self.DsetTest,
			                                batch_size  = self.a.batch_size,
			                                shuffle     = False,
			                                sampler     = self.DsamplerTest,
			                                num_workers = 0,
			                                pin_memory  = False)
		return self
	
	def load(self, path):
		self.S = torch.load(os.path.join(path, "snapshot.pkl"))
		return self
	
	def dump(self, path):
		self.S.zvitStepNum = self.z.globalStep
		torch.save(self.S,  os.path.join(path, "snapshot.pkl"))
		return self
	
	@staticmethod
	def download(a):
		"""
		Download the dataset or datasets required.
		"""
		if a.dataset in {"all", "mnist"}:
			MNIST   (a.dataDir, True,    download=True)
			MNIST   (a.dataDir, False,   download=True)
		if a.dataset in {"all", "cifar10"}:
			CIFAR10 (a.dataDir, True,    download=True)
			CIFAR10 (a.dataDir, False,   download=True)
		if a.dataset in {"all", "cifar100"}:
			CIFAR100(a.dataDir, True,    download=True)
			CIFAR100(a.dataDir, False,   download=True)
		if a.dataset in {"all", "svhn"}:
			SVHN    (a.dataDir, "train", download=True)
			SVHN    (a.dataDir, "extra", download=True)
			SVHN    (a.dataDir, "test",  download=True)
		return 0
	
	@property
	def name(self):
		"""A unique name containing every attribute that distinguishes this
		experiment from another and no attribute that does not."""
		attrs = [
			self.a.seed,
			self.a.model,
			self.a.dataset,
			self.a.dropout,
			self.a.num_epochs,
			self.a.batch_size,
			self.a.cuda,
			self.a.optimizer,
			self.a.lr,
			self.a.clipnorm,
			self.a.clipval,
			self.a.l1,
			self.a.l2,
			self.a.fastdebug,
		]
		return "-".join([str(s) for s in attrs]).replace("/", "_")
	@property
	def uuid(self):
		u = nauka.utils.pbkdf2int(128, self.name)
		u = uuid.UUID(int=u)
		return str(u)
	@property
	def dataDir(self):
		return self.a.dataDir
	@property
	def logDir(self):
		return os.path.join(self.workDir, "logs")
	@property
	def isDone(self):
		return (self.S.epochNum >= self.a.num_epochs or
		       (self.a.fastdebug and self.S.epochNum >= self.a.fastdebug))
	@property
	def exitcode(self):
		return 0 if self.isDone else 1
