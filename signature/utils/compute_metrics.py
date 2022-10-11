import numpy as np
import ot
import ot.sliced

METRIC = "euclidean"

def metrics(ns, seeds, location_template, true_samples, thin=1, sliced=True):

	"""
	ns and seeds are iterables containing the budgets and seeds to compute
	metrics for.
	location_template should be a string which will be formatted with {0} and
	{1} corresponding to the seed and budget, respectively.
	true_samples should be of shape (n_samples, dim)
	"""

	swds, meandists = dict(), dict()
	true_mean = np.mean(true_samples, axis=0)

	for n in ns:
		for seed in seeds:
			print()
			print(n, seed)
			try:
				samples = np.loadtxt(location_template.format(seed, n))
				if thin != 1:
					samples = samples[::thin]
				print("Sample shape: ", samples.shape)
				if sliced:
					swd = ot.sliced.sliced_wasserstein_distance(samples, true_samples, 
															n_projections=2000)
				else:
					M = ot.dist(samples, true_samples, metric=METRIC)
					gamma, log = ot.emd([], [], M, log=True)
					swd = log["cost"]
				try:
					swds[n].append(swd)
				except KeyError:
					swds[n] = [swd]
				meandist = np.sum((np.mean(samples, axis=0) - true_mean)**2)
				try:
					meandists[n].append(meandist)
				except KeyError:
					meandists[n] = [meandist]
				print(swd, meandist)
			except FileNotFoundError:
				print("Missing file at seed", seed, "and budget", n)

	return swds, ns, meandists
