import numpy as np
from sbi import utils
import torch
import torch.distributions as tds

class UniformTriangle:

	def __init__(self, numpy=False):
		self.A = np.array([-2.,1.])
		self.B = np.array([0.,-1.])
		self.C = np.array([2.,1.])
		self.v = np.stack([self.A, self.B, self.C])
		assert self.v.shape == (3, 2)
		self.to_numpy = numpy

	def sample(self, n_samples=1):
		if isinstance(n_samples, tuple):
			if len(n_samples) > 0:
				n_samples = n_samples[0]
			else:
				n_samples = 1
		x = np.sort(np.random.rand(2, n_samples), axis=0)
		col = np.column_stack([x[0], x[1]-x[0], 1.-x[1]])
		p = col @ self.v
		if self.to_numpy:
			if n_samples == 1:
				p = p[0]
			return p
		return torch.as_tensor(p).float()

	def log_prob(self, th):
		if type(th) is np.ndarray:
			th = torch.from_numpy(th)
		if len(th.size()) == 1:
			th1, th2 = th[0], th[1]
		else:
			th1, th2 = torch.as_tensor(th[:,0]), torch.as_tensor(th[:,1])
		mask = (th1.gt(-2.) * th1.lt(2.)
			* th2.gt(-1 - th1) * th2.gt(th1 - 1)
			* th2.lt(1) * th2.gt(-1))
		if len(th.shape) == 1:
			logprob = 0. if mask else -float("inf")
		else:
			logprob = [0. if x else -float("inf") for x in mask]
		if self.to_numpy:
			return np.array(logprob)
		return torch.tensor(logprob)

class BoxUniform:

	def __init__(self, low, high, numpy=False):

		self.bu = utils.BoxUniform(low=low, high=high)
		self.to_numpy = numpy

	def sample(self, *args, **kwargs):

		samples = self.bu.sample(*args, **kwargs)
		if self.to_numpy:
			samples = samples.numpy()
		return samples

	def log_prob(self, theta, **kwargs):

		if type(theta) is np.ndarray:
			theta = torch.from_numpy(theta)
		log_probs = self.bu.log_prob(theta, **kwargs)
		if self.to_numpy:
			log_probs = log_probs.numpy()
		return log_probs

class BoxGamma:

	def __init__(self, lmbdas, nus, numpy=False):

		assert len(lmbdas) == 2, "This is not a general class, only for specific prior for GSE model"
		self.lmbda_bet, self.lmbda_gam = lmbdas
		self.nu_bet, self.nu_gam = nus
		self.beta_prior = tds.gamma.Gamma(self.lmbda_bet, self.nu_bet, validate_args=False)
		self.gamma_prior = tds.gamma.Gamma(self.lmbda_gam, self.nu_gam, validate_args=False)
		self.to_numpy = numpy

	def sample(self, n_samples=1):

		if isinstance(n_samples, int):
			n_samples = (n_samples,)
		beta_sample = self.beta_prior.sample(n_samples)
		gamma_sample = self.gamma_prior.sample(n_samples)
		p = torch.stack((beta_sample, gamma_sample)).T
		if self.to_numpy:
			if n_samples == (1,):
				p = p[0]
			p = p.numpy()
		return p

	def log_prob(self, th):

		if len(th.shape) == 2:
			th0, th1 = th[:,0], th[:,1]
			mask = (th0 > 0.) * (th1 > 0.)
		elif len(th.shape) == 1:
			th0, th1 = float(th[0]), float(th[1])
			mask = torch.tensor([th0 > 0., th1 > 0.])
		else:
			raise IndexError("This class is only for 2D Gamma prior for GSE model")
		th0, th1 = torch.as_tensor(th0), torch.as_tensor(th1)
		vals = (self.beta_prior.log_prob(th0) + self.gamma_prior.log_prob(th1)).reshape(-1)
		new_vals = torch.tensor([val if m else -float("inf") for val, m in zip(vals, mask)])
		if self.to_numpy:
			new_vals = new_vals.numpy()
		return new_vals
