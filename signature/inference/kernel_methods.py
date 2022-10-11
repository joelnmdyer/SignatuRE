from hyperopt import fmin, hp, tpe, Trials
import logging
from numba import njit
import numpy as np
from scipy.linalg import svd
import sigkernel
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import kernels
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import StratifiedKFold
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.class_weight import compute_class_weight
import torch
import warnings

warnings.filterwarnings('ignore')

#########################
#  Code for SignatuRE   #
#########################

class CustomLR:

	def __init__(self, mapping, **kwargs):
		self.mapping = mapping
		self.lr = LogisticRegression(**kwargs)
		self._xkern = None
		self._mapping = None

	def fit(self, X, y):
		X = np.dot(X, self.mapping)
		self.lr.fit(X, y)

	def set_xkern(self, xkern):
		"""
		xkern needs to be of shape (n_components, 1)
		"""
		self._xkern = xkern
		self._mapping = xkern*self.mapping

	def predict_proba(self, X):
		if not (self._xkern is None):
			X = np.dot(X, self._mapping)
		else:
			X = np.dot(X, self.mapping)
		return self.lr.predict_proba(X)

def get_mapping(gram):
	U, S, V = svd(gram)
	S = np.maximum(S, 1e-12)
	return np.dot(U / np.sqrt(S), V)

#########################
# Code for K2-RE method #
#########################

@njit
def _gauss_rbf(xi, xj, c):
	diff = xi-xj
	dot_diff = np.sum(diff**2)
	return np.exp(-dot_diff/c)

@njit
def mmd_est(x, y, c):
	"""
	Function for estimating the MMD between samples x and y using Gaussian RBF
	with scale c.
	Args:
		x (np.ndarray): (n_samples, n_dims) samples from first distribution.
		y (np.ndarray): (n_samples, n_dims) samples from second distribution.
	Returns:
		float: The mmd estimate."""

	n_x = x.shape[0]
	n_y = y.shape[0]

	factor1 = 0.
	for i in range(n_x):
		for j in range(n_x):
			if (j == i): continue
			factor1 += _gauss_rbf(x[i:i+1], x[j:j+1], c)
	factor1 /= (n_x*(n_x-1))

	factor2 = 0.
	for i in range(n_y):
		for j in range(n_y):
			if (j == i): continue
			factor2 += _gauss_rbf(y[i:i+1], y[j:j+1], c)
	factor2 /= (n_y*(n_y-1))

	factor3 = 0.
	for i in range(n_x):
		for j in range(n_y):
			factor3 += _gauss_rbf(x[i:i+1], y[j:j+1], c)
	factor3 *= 2/(n_x*n_y)

	return factor1 + factor2 - factor3

@njit
def _compute_mmd_matrix(mmd_matrix, xs, ys, scale, sym=False):

	batchx, batchy = xs.shape[0], ys.shape[0]
	denom = 2*scale**2
	for i in range(batchx):
		idx = 0
		if sym:
			idx = i
		for j in range(idx, batchy):
			mmd_val = mmd_est(xs[i], ys[j], denom)
			mmd_matrix[i,j] = mmd_val
			if sym:
				mmd_matrix[j,i] = mmd_val
	return mmd_matrix

class K2:

	def __init__(self, base_scale, outer_scale=None):
		"""scale enters into the base kernel as exp(-|x - y|^2/(2*scale**2)),
		which is used to compute the MMD estimate
		"""
		self.base_scale = base_scale
		self.outer_scale = outer_scale

	def compute_mmd_matrix(self, xs, ys, sym=False):
		batchx, batchy = xs.shape[0], ys.shape[0]
		mmd_matrix = np.empty((batchx, batchy))
		mmd_matrix = _compute_mmd_matrix(mmd_matrix, xs, ys, self.base_scale, sym=False)
		return mmd_matrix

	def compute_gram(self, xs, ys, sym=False):
		mmd_matrix = self.compute_mmd_matrix(xs, ys, sym=sym)
		return np.exp( -mmd_matrix/self.outer_scale )


##############################################
# General-purpose loss function for hyperopt #
##############################################
N_SPLITS = 5
MAX_ITER = 500
@ignore_warnings(category=ConvergenceWarning)
def _loss_function(x, method, n_components, n_levels, K, xx_train, X_train, mmd_matrix, tt_train, y_train, class_weight):
	"""Assumes x = [length_scale1, length_scale2, ..., length_scaleN, kernel_scale, regularisation_par]"""
	elif method == "signature":
		# Parse input
		length_scales = x[:-1]	
	elif method == "k2":
		length_scales = x[:-2]
		# TODO: Compute X kernel with kernel_scale as hyperparameter. Should be shape (N, n_components)
		X_train = np.exp( -mmd_matrix/x[-2] )
	rbf = kernels.RBF(length_scale=length_scales)
	T = rbf(tt_train, tt_train[:int(n_components), :])
	G = X_train * T
	mapping = get_mapping(G[:int(n_components), :int(n_components)])
	feature = G[:, :int(n_components)].dot(mapping)
	loss = 0
	zeros, ones = np.bincount(y_train)
	skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=False, random_state=None)
	for train_idx, test_idx in skf.split(feature, y_train):
		G_ = feature[train_idx]
		_G = feature[test_idx]
		y_train_ = y_train[train_idx]
		_y_train = y_train[test_idx]
		clf = LogisticRegression(max_iter=MAX_ITER, C=1./x[-1], fit_intercept=True, class_weight=class_weight)
		clf.fit(G_, y_train_)
		_y_pred = clf.predict_proba(_G)[: ,1]
		loss += log_loss(_y_train, _y_pred, sample_weight=np.array([1./(2*zeros) if y == 0 else 1./(2*ones) for y in _y_train]))
	loss /= N_SPLITS
	return loss

def create_theta_kern(tt_train, n_components, sigma_th):

	rbf = kernels.RBF(length_scale=sigma_th)
	def theta_kern(theta):
		if len(theta.shape) == 1:
			theta = theta.reshape(1,-1)
		T_test = rbf(theta, tt_train[:n_components, :])
		return T_test

	return theta_kern

#######################
# Script for training #
#######################
def train_kernel_classifier(prior, simulator, obs, L, K, n_components_raw, task, method):

	n_components = n_components_raw*(K+1)
	print("Using {0} components".format(n_components))

	# Initial training set: to be augmented with confused samples
	t_train = np.stack([prior.sample() for i in range(L)])
	x_train = [simulator(theta) for theta in t_train]
	x_train = np.stack(x_train)
	scale = (x_train.max() - x_train.min())
	print("Normalise x with {0}".format(scale))
	logging.info("Normalise all x with {0}".format(scale))
	normobs = np.expand_dims((obs / scale), axis=0)
	# This is ok because GSE already comes rescaled
	x_train = x_train / scale

	# Augment with confused samples
	thetas, ys = [], []
	for i in range(L):
		thetas.append(t_train[i])
		ys.append(1)
		for j in range(K):
			thetas.append(prior.sample())
			ys.append(0)

	tt_train = np.stack(thetas)
	y_train = np.array(ys)
	class_weight = compute_class_weight('balanced', np.unique(y_train), y_train)
	logging.info("x_train, t_train, y_train shapes: {0}, {1}, {2}".format(x_train.shape, t_train.shape, y_train.shape))

	# For optimisation of kernel parameters
	space = {"th{0}".format(i):hp.loguniform("th{0}".format(i), np.log(1e-3), np.log(1e3))
			 for i in range(tt_train.shape[1])}
	space["reg"] = hp.loguniform("reg", np.log(1e-5), np.log(1e4))

	logging.info("x_train, t_train, y_train shapes: {0}, {1}, {2}".format(x_train.shape, t_train.shape, y_train.shape))

	ll = False
	ADD_TIME = False
	if method in ["signature"]:
		if task not in ["GSE"]:
			normobs = np.expand_dims(normobs, axis=-1)
			ADD_TIME = True
			x_train = np.expand_dims(x_train, axis=-1)

	# Sigkernel things:
	x0 = sigkernel.transform(normobs, at=ADD_TIME, ll=ll, scale=1.)
	# Add time and rescale time series – all methods
	xx_train = sigkernel.transform(x_train, at=ADD_TIME, ll=ll, scale=1.)

	if method == "signature":

		# Untruncated signature kernel
		sigma = np.median(euclidean_distances(x0[0,:,:]))
		x0 = torch.as_tensor(x0)
		static_kernel = sigkernel.RBFKernel(sigma=sigma)
		signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order=1)
		# Add time and rescale time series
		xx_train = torch.tensor(xx_train)
		print("Computing signature Gram matrix...")
		X_train = signature_kernel.compute_Gram(xx_train, xx_train[:n_components_raw,:,:]).cpu().numpy()
		X_train = np.repeat(np.repeat(X_train, K+1, axis=0), K+1, axis=1)
		print("Signature Gram =", X_train)

		def loss_function(x):
			"Convert the input dictionary into np.array to feed into _loss_function"
			x = np.array([x["th{0}".format(i)] for i in range(len(x)-1)] + [x["reg"]])
			return _loss_function(x, method, n_components, None, K, xx_train, X_train, None,
								  tt_train, y_train, class_weight)

	elif method == "k2":

		if (task == "GSE"):
			obs = obs[:, :-1]
			x_train = x_train[:, :, :-1]
		else:
			x_train = np.expand_dims(x_train, axis=-1)
			obs = obs.reshape(-1,1)
		x0 = np.expand_dims(obs, axis=0)

		# Median heuristic for Gaussian base kernel
		sigma = np.median(np.sqrt(euclidean_distances(x0[0, :, :])))
		print(sigma)
		k2 = K2(base_scale=sigma, outer_scale=None)

		print("Computing K2 Gram matrix...")
		mmd_matrix = k2.compute_mmd_matrix(x_train, x_train[:n_components_raw,:,:])
		mmd_matrix = np.repeat(np.repeat(mmd_matrix, K+1, axis=0), K+1, axis=1)
		print("MMD matrix =", mmd_matrix)

		space["oscale"] = hp.loguniform("oscale", np.log(1e-3), np.log(1e3))

		def loss_function(x):
			"Convert the input dictionary into np.array to feed into _loss_function"
			x = np.array([x["th{0}".format(i)] for i in range(len(x)-2)] + [x["oscale"], x["reg"]])
			return _loss_function(x, method, n_components, None, K, None, None, mmd_matrix,
								  tt_train, y_train, class_weight)

	# Optimise hyperparameters
	trials = Trials()
	best_params = fmin(fn=loss_function, space=space, max_evals=150, algo=tpe.suggest,
					   trials=trials)
	
	elif method == "k2":
		# TODO: Compute X_train with optimal outer kernel scale parameter
		X_train = np.exp( -mmd_matrix/best_params["oscale"] )
		k2 = K2(base_scale=sigma, outer_scale=best_params["oscale"])

	###########################
	# Train kernel classifier #
	###########################
	print("Retraining on full data with optimal hyperparameters...")
	sigma_th = np.array([best_params["th{0}".format(i)] for i in range(tt_train.shape[1])])
	logging.info("RBF hyperparameters: {0} for x; {1} for theta".format(sigma, sigma_th))
	logging.info("Regularisation parameter = {0}".format(1./best_params["reg"]))
	rbf = kernels.RBF(length_scale=sigma_th)
	T_train = rbf(tt_train, tt_train[:int(n_components), :])
	print("Gram matrix computed!")
	G_train = X_train * T_train
	mapping = get_mapping(G_train[:int(n_components), :int(n_components)])

	clf = CustomLR(mapping, max_iter=MAX_ITER, C=1./best_params["reg"], fit_intercept=True, class_weight=class_weight)
	clf.fit(G_train[:, :int(n_components)], y_train)

	elif method == "signature":
		def compute_unexpanded(x):
			return signature_kernel.compute_Gram(x, xx_train[:n_components_raw, :, :]).cpu().numpy()
	elif method == "k2":
		def compute_unexpanded(x):
			return k2.compute_gram(x, x_train[:n_components_raw, :, :])

	def inn_prods(x):
		unexpanded = compute_unexpanded(x)
		expanded = np.expand_dims(np.repeat(unexpanded, K+1), axis=0)
		return expanded

	logging.info("Best cross-entropy loss score: {0}".format(min([t["result"]["loss"] for t in trials])))
	theta_kern = create_theta_kern(tt_train, int(n_components), sigma_th)

	return clf, x0, prior, inn_prods, theta_kern
