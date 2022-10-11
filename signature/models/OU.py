from collections import namedtuple
from numba import njit
import numpy as np
import os
import scipy.stats
import statsmodels.tsa as sttsa
import torch
from tqdm import trange

from signature.utils import distributions

loc=os.path.abspath(__file__)
basedir=os.path.dirname(loc)
fullpath=os.path.join(basedir, "../data/OU_obs_05_1_seed0.txt")
data = np.loadtxt(fullpath)

RParam = namedtuple("RParam", ["th1", "th2", "x0", "dt", "T"])
default_param = RParam(
	th1 = 0.5,
	th2 = 1.,
	x0 = 10,
	dt = 0.2,
	T = 50
)

lbs = [0.,-2.]
ubs = [1.,2.]
prior = distributions.BoxUniform(low=torch.tensor(lbs), high=torch.tensor(ubs), numpy=True)
torch_prior = distributions.BoxUniform(low=torch.tensor(lbs), high=torch.tensor(ubs))
n_pars = len(lbs)

def autocorr_lags2(x):
	return sttsa.stattools.acf(x, nlags=2)[1:]

def mean(x):
	return np.mean(np.array(x))

def std(x):
	return np.std(np.array(x))

statistics = [mean, std, autocorr_lags2]

@njit 
def _simulate(model, T, x, seed=None):
	if seed is not None:
		np.random.seed(seed)

	expth2 = np.exp(model.th2)

	for t in range(1, x.size):
		dx = model.dt*(model.th1*(expth2 - x[t-1]) + 0.5*np.random.normal())
		x[t] = x[t-1] + dx

class Model:
	def __init__(self, T=50, pars=default_param):
		self.pars = pars
		self.T = T
		self.x = np.zeros(T+1)

	def simulate(self, pars=None, seed=None):
		if pars is not None:
			x0, T, dt = self.pars.x0, self.pars.T, self.pars.dt
			self.pars = RParam(th1=float(pars[0]), th2=float(pars[1]),
							   x0=x0, dt=dt, T=T)

		self.x[0] = self.pars.x0
		_simulate(self.pars, self.T, self.x, seed)
		return self.x.copy()


def loglike(y, th):

	th1, th2 = th
	x0, T, dt = default_param.x0, default_param.T, default_param.dt
	ll = 0
	norm_logpdf = scipy.stats.norm.logpdf
	expth2 = np.exp(th2)
	center = (th1 * expth2 * dt)
	coeff =  (1 - (th1 * dt))
	std = dt/2.
	for t in range(1, len(y)):
		# It's just AR(1)
		ll += norm_logpdf(y[t], center + y[t-1]*coeff, std)
	return ll


def sample_from_post(y, n_samples=10_000, x0=None, cov=np.eye(2),
					 seed=1, sigma=1.):

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
	rev_logpost = loglike(y,x_) + prior.log_prob(torch.tensor(x_).float())

	test_output = 0.
	acceptance_rate = 0.
	neg_inf = float("-inf")

	t = trange(1, n_samples, position=0, leave=True)
	for n in t:
		# Propose new point
		x = proposal.rvs(mean=x_, cov=cov)
		new_logpost = loglike(y,x) + prior.log_prob(torch.tensor(x).float())
		# Reject if outside prior range
		if new_logpost == neg_inf:
			test_output += 1
			xs[:, n] = x_
			continue
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
