from collections import namedtuple
from numba import njit
import numpy as np
import os
import scipy.stats
import torch
from torch.distributions import gamma

from signature.utils import distributions

RParam = namedtuple("RParam", ["beta", "gamma", "max_time", "n", "i0"])
default_param = RParam(beta=1e-2, gamma=1e-1, max_time=50, n=10**2, i0=1) 


basedir = os.path.dirname(os.path.abspath(__file__))
obs_fname = os.path.join(basedir, "../data/GSE_obs.txt")
data = np.loadtxt(obs_fname)
lbs=[0,0]
ubs=[None,None]
lambdas, nus = [0.1, 0.2], [2, 0.5]
prior = distributions.BoxGamma(lambdas, nus, numpy=True)
torch_prior = distributions.BoxGamma(lambdas, nus)
n_pars = len(lbs)

@njit
def _step(model, ns, ni, nr, t, seed):

	if t == 0:	
		if not (seed is None):
			np.random.seed(seed)

	p_infection = model.beta * ns * ni
	p_recovery = model.gamma * ni
	p_total = p_infection + p_recovery

	dt = np.random.exponential(1 / p_total)
	t += dt

	if np.random.rand() < p_infection / p_total:
		ns -= 1
		ni += 1

	else:
		ni -= 1
		nr += 1
	return ns, ni, nr, t

#@njit
def _simulate(model, nss, nis, nrs, xy_int, y_int, niend, nrend, seed):

	t = 0
	ts = [0]

	ns, ni, nr = nss[0], nis[0], nrs[0]
	while (t < model.max_time) and (ni > 0):

		# Update state
		new_ns, new_ni, new_nr, new_t = _step(model, ns, ni, nr, t, seed)
		if new_t > model.max_time:
			ts.append(model.max_time)	
			nss.append(ns)
			nis.append(ni)
			nrs.append(nr)
			break
		ts.append(new_t)

		ns, ni, nr = new_ns, new_ni, new_nr
		# Store variables
		nss.append(ns)
		nis.append(ni)
		nrs.append(nr)

		# Update state if you didn't do it above
		t = new_t

	if (t < model.max_time) and (ni == 0):
		# If we get to here, then the simulation didn't end but there are no
		# more infected individuals. In that case, you need to observe the
		# same state for the remaining time steps
		ts.append(model.max_time)
		nss.append(ns)
		nis.append(ni)
		nrs.append(nr)
		# We also don't need to worry about updating the integrals, because
		# the integrands are zero if ni == 0

	# Want to return integrals and TOTAL number infected and recovered
	# throughout pandemic. R is absorbing so nr is fine for latter
	return xy_int, y_int, ts

class Model:

	"""
	Input:
	- pars:		RParam instance containing parameter values
	"""

	def __init__(self, pars=default_param):
		self.pars = pars
		self.reset()

	def reset(self):
		self.ni = self.pars.i0
		self.ns = self.pars.n - self.pars.i0
		self.nr = 0
		self.xy_int = 0.
		self.y_int = 0.

	def simulate(self, pars=None, seed=None):

		"""
		This function runs the epidemic simulation. Returns state of simulation at intervals of 0.5 days
		Input:
			pars: epidemic parameters, pars[0] = beta, pars[1] = gamma
			seed (int): for initialising RNG
		
		"""

		if not (pars is None):
			self.pars = RParam(beta=float(pars[0]), gamma=float(pars[1]),
							   n=default_param.n, i0=default_param.i0,
							   max_time=default_param.max_time)

		self.reset()

		self.nss, self.nis, self.nrs = [], [], []
		self.nss.append(self.ns)
		self.nis.append(self.ni)
		self.nrs.append(self.nr)
		self.xy_int, self.y_int, ts_ = _simulate(self.pars, self.nss, self.nis,
											self.nrs, self.xy_int,
											self.y_int, self.ni, self.nr,
											seed)
		ts_ = np.array(ts_)
		self.ts_ = ts_
		# Observations
		nis, nrs, ts = [self.nis[0]/self.pars.n], [self.nrs[0]/self.pars.n], [0.]
		Dt = 0.5
		for i in range(1, int(self.pars.max_time/Dt)+1):
			t = Dt*i
			ts.append(t/self.pars.max_time)
			# Find state as it is at t. Do so by finding the first event after t and
			# copying the state immediately before that
			idx = np.min(np.where(ts_ >= t)) - 1
			nis.append(self.nis[idx]/self.pars.n)
			nrs.append(self.nrs[idx]/self.pars.n)
		return np.stack((nis, nrs, ts)).T
