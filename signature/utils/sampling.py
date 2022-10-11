import numpy as np
import scipy.stats
import torch
from tqdm import tqdm

def mh(log_prob, d, start, method, cov=None, scale=None, n_samples=100_000,
	   seed=None):

	"""
	Input:
	- log_prob:			function taking in theta of shape (1,d) or (d,) and
						returning estimate of posterior density evaluated at
						the observation and theta
	- d:				int, dimension of parameter
	- start:			np.array of shape (1,d) or (d,), indicating the
						initial value of the random walk
	"""

	to_torch = not (method in ["signature", "k2"])

	if cov is None:
		cov = np.eye(d)
	if scale is None:
		scale = 2/np.sqrt(d)
	cov = cov * scale
	if not (seed is None):
		np.random.seed(seed)

	# Innovations for random walk
	q = scipy.stats.multivariate_normal
	es = scipy.stats.multivariate_normal.rvs(np.zeros(d), cov, size=n_samples)
	ps = np.random.random(n_samples)

	iterator = tqdm(range(n_samples), position=0)
	n_test, n_acc = 0, 0

	samples = np.empty((n_samples, d))
	samples[0,:] = start
	th0 = start

	# if not to_torch:
	lp0 = log_prob(th0)
	# else:
	# 	lp0 = log_prob(torch.from_numpy(th0.reshape(-1).astype(np.float32)))
	for t in iterator:
		# Propose new sample
		th1 = th0 + es[t]
		if (th1.size != d):
			raise ValueError("Parameter of wrong dimension")
		# Evaluate log-probability density of posterior density at proposed
		# if not to_torch:
		lp1 = log_prob(th1)	
		if lp1 == -float("inf"):
			n_test += 1
			samples[t,:] = th0
			continue
		else:
			d_log_probs = lp1 - lp0
		# else:
			# lp1 = log_prob(torch.from_numpy(th1.reshape(-1).astype(np.float32)))
			# d_log_probs = (lp1 - lp0)
		# Compute log of acceptance probability
		loga = min([0, d_log_probs + 
					   q.logpdf(th0, mean=th1.reshape(-1), cov=cov) -
					   q.logpdf(th1, mean=th0.reshape(-1), cov=cov)])
		# Determine whether to accept or reject
		if loga == -float("inf"):
			n_test += 1
			samples[t,:] = th0
		else:
			if np.log(ps[t]) >= loga:
				# Reject th1
				samples[t,:] = th0
			else:
				# Accept th1
				samples[t,:] = th1
				n_acc += 1
				th0 = th1
				lp0 = lp1
		iterator.set_postfix({"Acc. rate":n_acc/(t+1), "Test":n_test/(t+1)})
		iterator.update()

	return samples

def sir(prior, ratio_estimator, n_samples1=100_000, n_samples2=1_000):

	"""
	Sampling importance resampling.
	"""

	samples = []
	for n in range(n_samples1):
		theta = prior.sample()
		try:
			theta = theta.numpy()
		except:
			pass
		samples.append(theta)
	importances = np.array([ratio_estimator(sample) for sample in samples])
	importances /= np.sum(importances)
	print(importances, importances.shape)
	idx = np.random.choice(np.arange(n_samples1), p=importances.reshape(-1), size=n_samples2)
	resamples = np.stack(samples)[list(idx)]
	return resamples
