import argparse
import logging
import numpy as np
import os
import sbi.utils as utils
from sbi.inference.base import infer
from sbi import analysis as analysis
from sbi.inference import SMCABC, SNRE_A, simulate_for_sbi, prepare_for_sbi
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import time
import torch

# Custom scripts/modules/packages
from signature.inference import kernel_methods
from signature.utils import networks
from signature.utils import io, sampling


def train_clf(task, method, L, K=2, n_components_raw=100, seed=0):

	"""
	Trains a binary classifier with method <method> to distinguish between
	samples (x, theta) from the joint distribution p(x, theta) and from the
	product of the marginals p(x)p(theta) associated with <task>.
	Input:
	- task:			str, name of model to run inference on, must be recognised
					by function get_task above.
	- method:		str, name of classifier to use, either "signature" or
					"gru-resnet"
	- L:			int, number of training examples (simulations) to generate
	- K:			int, number of contrasting examples. Only used when
					method == "signature"
	- seed:			int, seed for random number generator
	"""

	prior, sbi_prior, obs, simulator = io.get_task(task)

	if method in ["signature", "k2"]:

		clf, x0, _, inn_prods, theta_kern = kernel_methods.train_kernel_classifier(prior,
																				   simulator,
																				   obs,
																				   L,
																				   K,
																				   n_components_raw,
																				   task,
																				   method)

	elif method[:10] == "gru-resnet":
	
		IDIM = 1
		def sbi_simulator(x):
			return simulator(x)
		if task == "GSE":
			obs = obs[:, :-1]
			IDIM = 2
			# Remove time indices from GSE output
			def sbi_simulator(x):
				return simulator(x)[:,:-1]
		ODIM = 3
		if method != "gru-resnet":
			ODIM = eval(method[10:])
		simulator_wrapper, _prior = prepare_for_sbi(sbi_simulator, sbi_prior)

		# Instantiate the neural density ratio estimator
		embedding_net = networks.GRU(input_dim=IDIM, hidden_dim=32, num_layers=2,
							output_dim=ODIM)
		n_pars_embedding = sum(p.numel() for p in embedding_net.parameters() if p.requires_grad)
		logging.info("Embedding net has {0} parameters".format(n_pars_embedding))
		classifier = utils.get_nn_models.classifier_nn('resnet',
													   embedding_net_x=embedding_net)

		# Setup the inference procedure with the SNRE-A procedure
		inference = SNRE_A(prior=_prior, classifier=classifier)

		# Run the inference procedure on one round and L simulated data points
		theta, x = simulate_for_sbi(simulator_wrapper, _prior, num_simulations=L)
		if task not in ["GSE"]:
			x = x.unsqueeze(-1)
		elif task == "GSE":
			# Print this out to see that it gives you everything in the right place
			x = x.reshape(L, -1, 2)
		density_estimator = inference.append_simulations(theta, x).train()
		posterior = inference.build_posterior(density_estimator)
		posterior.set_default_x(obs.reshape(1,-1,IDIM))

		clf = posterior
		inn_prods = None
		theta_kern = None
		x0 = obs
		prior = _prior

	elif method in ["hc", "smcabc"]:

		def slope_intercept(data):
			reg = LinearRegression().fit(data[:-1].reshape(-1,1), data[1:].reshape(-1,1))
			slope = reg.coef_
			intercept = reg.intercept_
			return slope, intercept

		if task == "OU":			
			def summarise(data):	
				slope, intercept = slope_intercept(data)
				summary = np.array([np.mean(data), slope[0,0], intercept[0]])
				return summary

		elif task == "MA2":
			def summarise(data):
				var = np.var(data)
				rhos = sm.tsa.acf(data, nlags=2)[1:]
				return np.array([var, rhos[0], rhos[1]])
	
		elif task == "GSE":
			def summarise(data):
				data = data[:, :-1]
				N = data.shape[0]
				x, y = data[:,0], data[:,1]
				xmean = np.mean(x)
				ymean = np.mean(y)
				xvar = np.var(x, ddof=1)
				yvar = np.var(y, ddof=1)
				if xvar == 0.:
					xvar = 1e-30
				if yvar == 0.:
					yvar = 1e-30
				x, y = (x - xmean)/np.sqrt(xvar), (y - ymean)/np.sqrt(yvar)
				acx, acy = [], []
				for lag in [1,2]:
					acx.append(np.dot(x[:-lag], x[lag:]) / (N - 1))
					acy.append(np.dot(y[:-lag], y[lag:]) / (N - 1))
				ccxy = np.dot(x, y)/(N-1)
				summary = np.array([xmean, ymean, np.log(xvar + 1), np.log(yvar+1), ccxy] + acx + acy)
				return summary

		def sbi_simulator(x):
			data = simulator(x)
			return summarise(data)


		if method == "hc":

			x0 = summarise(obs)
			simulator_wrapper, _prior = prepare_for_sbi(sbi_simulator, sbi_prior)
			# Instantiate the neural density ratio estimator
			classifier = utils.get_nn_models.classifier_nn('resnet')

			# Setup the inference procedure with the SNRE-A procedure
			inference = SNRE_A(prior=_prior, classifier=classifier)

			# Run the inference procedure on one round and L simulated data points
			theta, x = simulate_for_sbi(simulator_wrapper, _prior, num_simulations=L)
			density_estimator = inference.append_simulations(theta, x).train()
			posterior = inference.build_posterior(density_estimator)
			posterior.set_default_x(x0)

			clf = posterior

		elif method == "smcabc":

			def _simulator(theta):
				return simulator(theta)[:, :-1].reshape(-1)

			print(_simulator(prior.sample()))

			simulator_wrapper, _prior = prepare_for_sbi(_simulator, sbi_prior)
			inference = SMCABC(simulator_wrapper, _prior, num_workers=20)
			clf = inference
			x0 = obs[:, :-1].reshape(-1)

		print(x0)
		inn_prods = None
		theta_kern = None
		prior = _prior

	return clf, x0, prior, inn_prods, theta_kern


def sample(method, clf, x0, start, sampling_method, n_samples=[50_000, 100_000], prior=None,
		  inn_prods=None, theta_kern=None):

	"""
	Uses a density ratio estimator clf to sample from the posterior for x0
	and prior.
	Inputs:
	- method:		str, either "signature" or "gru-resnet" depending on which
					classifier is being used
	- clf:			the density ratio estimator
	- x0:			the preprocessed observation
	- start:		np.array consisting of the start point for MCMC. Recommend
					using true parameter value that generated x0 for this
	- n_samples:	list of length 2 consisting of ints > 0. Trial run of MCMC
					uses n_samples[0] steps to estimate covariance matrix of
					Gaussian proposal density; proper run uses n_samples[1]
	- prior:		prior distribution, only used if method == "signature",
					otherwise ignored. Default None
	"""

	if method in ["signature", "k2"]:

		if prior is None:
			raise ValueError("Must provide prior for kernel classifier")

		def create_log_ratio_estimator(clf, x):
			"Create a ratio estimator from the signature-based classifier"
			X_test = inn_prods(x)
			clf.set_xkern(X_test.reshape(-1,1))

			lr = clf.lr
			coefficients = lr.coef_.T
			intercept = lr.intercept_
			vector = (clf._mapping).dot(coefficients)

			def log_ratio_estimator(theta):
				T_test = theta_kern(theta)
				return T_test.dot(vector) + intercept
			
			return log_ratio_estimator

		custom_log_ratio_estimator = create_log_ratio_estimator(clf, x0)
		custom_ratio_estimator = lambda theta: np.exp(custom_log_ratio_estimator(theta))

		def kernel_posterior(theta):
			"""
			Function to evaluate estimation of posterior density for
			kernel-based classifier.
			"""
			prior_logpdf = prior.log_prob(theta)
			if prior_logpdf == -float("inf"):
				return prior_logpdf
			else:
				log_weight = custom_log_ratio_estimator(theta)
				return log_weight + prior_logpdf

		log_post_prob = kernel_posterior

	elif (method[:10] == "gru-resnet") or (method == "hc"):

		def log_post_prob(th):
			# Convert th to torch.tensor
			th = torch.as_tensor(th).float()
			return clf.log_prob(th)

		# For sampling importance resampling
		custom_ratio_estimator = lambda th: float(torch.exp(clf.log_prob(th) - prior.log_prob(th)))

	elif method == "smcabc":

		samples = clf(x0, 1_000, 1_000, int(1e7), 0.8)
		return samples

	if sampling_method == "mh":
		# Pilot run to estimate covariance matrix of Gaussian proposal density
		samples = sampling.mh(log_post_prob, len(start), start, method,
							  n_samples=n_samples[0])
		cov = np.cov(samples.T)
		# Proper run
		samples = sampling.mh(log_post_prob, len(start), start, method,
							  n_samples=n_samples[1], cov=cov)
		samples = samples[::100]
	elif sampling_method == "sir":
		# SIR
		samples = sampling.sir(prior, custom_ratio_estimator, 50_000,
							   1_000)

	return samples


def train_inference(task, method, start, L, fname, K=2, sampling_method="mh",
					n_samples=[50_000, 100_000], seed=0, n_components_raw=100, start_time=0):

	print("Training classifier...")
	clf, x0, prior, s_kernel, t_kernel = train_clf(task, method, L, K=K,
												   n_components_raw=n_components_raw, seed=seed)
	logging.info("Training CPU time = {0}".format(time.process_time() - start_time))
	print("Sampling from posterior...")
	samples = sample(method, clf, x0, start, sampling_method, n_samples=n_samples, prior=prior,
					 inn_prods=s_kernel, theta_kern=t_kernel)
	print("Saving samples...")
	np.savetxt(fname, samples)
	print("Done.")


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Ratio estimation')
	parser.add_argument('--task', type=str,
						help='Name of task (simulator) to experiment with.')
	parser.add_argument('--method', type=str,
						help='Name of classification pipelines to use.')
	parser.add_argument('--L', type=int, nargs='+',
						help='Number of training simulations to use.')
	parser.add_argument('--K', type=int, default=1,
						help='Number of contrasting examples per simulation.')
	parser.add_argument('--s', type=str, default='mh',
						help="Sampling method in ['mh', 'sir'].")
	parser.add_argument('--n', type=int, default=100,
						help="Number of components retained in Nystrom DIVIDED BY (K+1).")
	parser.add_argument('--seed', type=int, nargs='+', help='Seeds for RNG.')
	args = parser.parse_args()

	if args.method == "sre":
		method = "signature"
	else:
		method = args.method

	if args.task == "OU":
		start = np.array([0.5, 1.])
	elif args.task == "MA2":
		start = np.array([0.6, 0.2])
	elif args.task == "GSE":
		start = np.array([1e-2, 1e-1])

	for L in args.L:

		for seed in args.seed:

			# Setup for saving output
			directory = "./{0}/{1}/".format(args.task, seed)
			if not os.path.exists(directory):
				os.makedirs(directory)
			if method in ["signature", "k2"]:
				fname = os.path.join(directory, "{0}_{1}_{2}_{3}_samples.txt".format(method, L, args.K, args.n))
				logging.basicConfig(filename=os.path.join(directory,
														  "{0}_{1}_{2}.log".format(method, L, args.K)),
									filemode="w", format="%(name)s - %(levelname)s - %(message)s",
									level=logging.INFO)
			else:	
				fname = os.path.join(directory, "{0}_{1}_samples.txt".format(method, L))
				logging.basicConfig(filename=os.path.join(directory,
														  "{0}_{1}.log".format(method, L)),
									filemode="w", format="%(name)s - %(levelname)s - %(message)s",
									level=logging.INFO)
			logging.info(args)
			logging.info("Seed = {0}".format(seed))

			# Run script
			start_time = time.process_time()
			train_inference(args.task, method, start, L, fname, sampling_method=args.s,
							K=args.K, seed=seed, n_components_raw=args.n, start_time=start_time)
			logging.info("Total CPU time = {0}".format(time.process_time() - start_time))
