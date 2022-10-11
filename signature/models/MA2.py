from collections import namedtuple
import os
from numba import njit
import numpy as np
import scipy.stats
import torch
from tqdm import trange

from signature.utils import distributions

loc=os.path.abspath(__file__)
basedir=os.path.dirname(loc)
fullpath=os.path.join(basedir, "../data/MA2_obs.txt")
data = np.loadtxt(fullpath)

RParam = namedtuple("RParam", ["th1", "th2"])
default_param = RParam(
	th1 = 0.6,
	th2 = 0.2
)

prior = distributions.UniformTriangle(numpy=True)
torch_prior = distributions.UniformTriangle()
n_pars = 2

@njit 
def _simulate(model, y, eps):

	y[1] = eps[1] + model.th1*eps[0]
	for t in range(2, y.size):
		y[t] = eps[t] + model.th1*eps[t-1] + model.th2*eps[t-2]	

class Model:
	def __init__(self, pars=default_param):
		self.pars = pars

	def simulate(self, pars=None, T=50, seed=None):
		if pars is not None:
			self.pars = RParam(th1=float(pars[0]), th2=float(pars[1]))
		self.y = np.zeros(T+1)
	
		if seed is not None:
			np.random.seed(seed)
		eps = np.random.normal(size=T+1)
		# Don't change first element of y from 0, because then this acts as
		# basepoint augmentation
		_simulate(self.pars, self.y, eps)
		# All non signature methods require removal of the first observation
		return self.y


def loglike(y, th):

	"""
	Input:
	- y:		observations 1,...,T
	- th:		iterable with elements [theta_1, theta_2]
	"""

	B = np.eye(y.size+1)
	B[1,0] = th[0]
	for i in range(2, B.shape[0]):
		B[i,i-2:i] = th[::-1]
	M = B.dot(B.T)[1:, 1:]
	dist = scipy.stats.multivariate_normal(mean=np.zeros(y.size), cov=M)
	return dist.logpdf(y)


def sample_from_post(y, n_samples=100_000, x0=None, cov=np.eye(2),
					 seed=1):

	"""
	For MCMC sampling from posterior
	"""

	np.random.seed(seed)

	if x0 is None:
		x0 = np.array([default_param.th1, default_param.th2])

	# Gaussian innovations
	proposal = scipy.stats.multivariate_normal

	xs = np.zeros((x0.size, n_samples))
	xs[:, 0] = x0

	x_ = x0
	rev_logpost = loglike(y,x_) + prior.log_prob(x_.reshape(1,-1))

	test_output = 0.
	acceptance_rate = 0.
	neg_inf = float("-inf")

	t = trange(1, n_samples, position=0, leave=True)
	for n in t:
		# Propose new point
		x = proposal.rvs(mean=x_, cov=cov)
		priorlogpdf = prior.log_prob(x.reshape(1,-1))
		# Reject if outside prior range
		if priorlogpdf == neg_inf:
			test_output += 1
			xs[:, n] = x_
			continue
		new_logpost = loglike(y,x) + priorlogpdf
		# Find log-pdf of new point from proposal
		new_logpdf = proposal.logpdf(x, mean=x_, cov=cov)
		# Find log-pdf of old point given new point
		rev_logpdf = proposal.logpdf(x_, mean=x, cov=cov)
		# Acceptance probability
		log_alpha = new_logpost + rev_logpdf - rev_logpost - new_logpdf
		if np.random.rand() >= np.exp(log_alpha):
			# Fail, reject proposal
			xs[:, n] = x_
			continue
		# Success
		xs[:, n] = x
		x_ = x 
		rev_logpost = new_logpost
		acceptance_rate += 1
		t.set_postfix({"Acc.:": acceptance_rate/n,
					   "test: ": test_output/n})
		t.refresh()  # to show immediately the update

	return xs
