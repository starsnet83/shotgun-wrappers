import numpy as np
import scipy.sparse as sparse
import ctypes
import os
from IPython import embed

# Load Shotgun library:
dir = os.path.dirname(__file__)
libraryPath = os.path.join(dir, 'shotgun_api.so')
lib = np.ctypeslib.load_library("shotgun_api.so", dir)
#ctypes.pydll.LoadLibrary("shotgunpy/shotgun_api.so")
#lib = ctypes.PyDLL("shotgun_api.so")

# Define result types:
lib.Shotgun_new.restype = ctypes.c_void_p
lib.Shotgun_run_lasso.restype = ctypes.POINTER(ctypes.c_double)
lib.Shotgun_run_lasso.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
lib.Shotgun_run_logreg.restype = ctypes.POINTER(ctypes.c_double)
lib.Shotgun_run_logreg.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
lib.Shotgun_set_A_sparse.restype = None
lib.Shotgun_set_A_sparse.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint), ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]
lib.Shotgun_set_A.restype = None
lib.Shotgun_set_A.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_long)]
lib.Shotgun_set_y.restype = None
lib.Shotgun_set_y.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
lib.Shotgun_set_lambda.restype = None
lib.Shotgun_set_lambda.argtypes = [ctypes.c_void_p, ctypes.c_double]
lib.Shotgun_set_threshold.restype = None
lib.Shotgun_set_threshold.argtypes = [ctypes.c_void_p, ctypes.c_double]
lib.Shotgun_set_num_threads.restype = None
lib.Shotgun_set_num_threads.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.Shotgun_set_use_offset.restype = None
lib.Shotgun_set_use_offset.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.Shotgun_set_initial_conditions.restype = None
lib.Shotgun_set_initial_conditions.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_double]

# Define Shotgun interface:
class ShotgunSolver(object):

	def __init__(self):
		self.obj = lib.Shotgun_new()
		self.set_num_threads(1)
		self.set_active_set_cutoff_ratio(0.9)
		
	def attach_A(self, A):
		"""Attaches Nxd data matrix to python object"""

		if (A.ndim != 2):
			raise Exception("A must be 2d")
		if (np.iscomplex(A).any()):
			raise Exception("Sorry, imaginary values are not supported")
		self.A = A
		self.d = A.shape[1]

	def load_A(self, A):
		"""Passes Nxd data matrix to C library, matrix may be sparse or dense"""

		(N, d) = A.shape
		if (sparse.issparse(A)):
			if (not sparse.isspmatrix_csc(A)):
				A = A.tocsc()
			# Sparse matrix, need to pass indices and values as separate arrays
			indicesArg = A.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint))
			dataArg = A.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
			indptrArg = A.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint))
			lib.Shotgun_set_A_sparse(self.obj, dataArg, indicesArg, indptrArg, ctypes.c_uint(A.nnz), ctypes.c_uint(N), ctypes.c_uint(d))
		else:
			# Dense matrix, pass entire matrix as a big array
			matrixArg = A.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
			lib.Shotgun_set_A(self.obj, matrixArg, ctypes.c_int(N), ctypes.c_int(d), A.ctypes.strides)

	def load_y(self, y):
		"""Passes Nx1 labels matrix to C library"""
		if (np.iscomplex(y).any()):
			raise Exception("Sorry, imaginary values are not supported")

		self.y = y.flatten()
		arrayArg = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
		lib.Shotgun_set_y(self.obj, arrayArg, ctypes.c_int(len(y))) 

	def set_lambda(self, value):
		"""Passes regularization parameter to C library"""
		if (not np.isreal(value) or value < 0):
			raise Exception("Lambda must be a nonnegative real value")
		self.lam = value
		lib.Shotgun_set_lambda(self.obj, ctypes.c_double(value))

	def set_tolerance(self, value):
		"""Passes stopping threshold to C library - algorithm completes
		when smallest change is less than this tolerance"""
		print self.obj
		lib.Shotgun_set_threshold(self.obj, ctypes.c_double(value))

	def set_use_offset(self, value):
		"""Set to True to use an unregularized offset, otherwise False.
		The default value is True."""
		lib.Shotgun_set_use_offset(self.obj, ctypes.c_int(value))

	def set_num_threads(self, value):
		"""Sets the maximum number of parallel updates to use in the
		coordinate descent algorithm."""
		value = int(value)
		self.numThreads = value
		lib.Shotgun_set_num_threads(self.obj, ctypes.c_int(value))

	def set_initial_conditions(self, (w, offset)):
		"""Specify a dx1 initial w vector and a scalar initial offset.
		Default values are all zeros."""
		wArg = w.ctypes.data_as(ctypes.POINTER(ctypes.c_double))	
		offsetArg = ctypes.c_double(offset)
		lib.Shotgun_set_initial_conditions(self.obj, wArg, offsetArg)

	def set_active_set_cutoff_ratio(self, value):
		"""Parameter for active sets, default is 0.9"""
		self.activeSetCutoffRatio = value

	def run_lasso(self, initialConditions=None):
		"""Solves lasso problem specified by A, y, and lambda with optional
		initial conditions.  Calls C library that implements parallel
		coordinate descent.  Uses an active set method (implemented in
		Python wrapper) to concentrate computation and improve runtimes.
		Returns a solution object with several attribuets."""

		if (self.y.shape[0] != self.A.shape[0]):
			raise Exception("A and y must have same number of training examples")	

		# Set up initial conditions and form residual vector:
		if (initialConditions):
			(w, offset) = initialConditions
			residuals = self.A * np.mat(w).T + offset - np.mat(self.y).T
			residuals = np.array(residuals).flatten()
		else:
			w = np.zeros(self.d) 
			offset = self.y.mean()
			residuals = self.y - offset

		# Store some values before main loop:	
		cutoff = self.activeSetCutoffRatio * self.lam
		maxThreads = self.numThreads
		obj = float('inf')

		while True:
			# Form active set based on correlation with current residual vector
			# (most correlated features are included, other features are ignored):
			subGrad = np.array(self.A.T * np.mat(residuals).T).flatten()
			corIndices = np.where(abs(subGrad) > cutoff)[0]
			nonzeroIndices = np.where(w != 0)[0]
			if (len(nonzeroIndices)):
				currentIndices = np.union1d(corIndices, nonzeroIndices)
			else:
				currentIndices = corIndices

			# Form active matrix:
			if (len(currentIndices) > 1 or sparse.issparse(self.A)):
				currentA = self.A[:, currentIndices]
			else:
				# avoid an issue with 1d np arrays here...
				currentA = np.mat(self.A[:, currentIndices]).T

			# Try to avoid diverging:
			current_d = currentA.shape[1]
			if (current_d < 250):
				self.set_num_threads(1)
			elif (current_d < 1000):
				self.set_num_threads(min(2, maxThreads))
			elif (current_d < 2500):
				self.set_num_threads(min(4, maxThreads))
			else:
				self.set_num_threads(maxThreads)

			# Form initial conditions:
			wInit = w[currentIndices]
			offsetInit = offset

			# Solve subproblem:
			sol = self.solve_lasso_subproblem(currentA, (wInit, offsetInit))

			# Return threads back to normal:
			self.set_num_threads(maxThreads)

			# Reform w vector of length d:	
			w = np.zeros(self.d)
			w[currentIndices] = sol.w

			# Check if done:
			if (abs(sol.obj - obj) < 1e-4):
				break
			else:
				obj = sol.obj
				residuals = np.array(sol.residuals).flatten()
				offset = sol.offset

		sol.w = w  # length d return sol
		return sol

	def solve_lasso_subproblem(self, A, init=None):
		"""Helper method that calls the C library to solve lasso.
		On the argument A.  Assumes y and lambda are already loaded."""

		# Load A into C library:
		self.load_A(A)	
		d = A.shape[1]

		# Load initial conditions:
		if (init):
			self.set_initial_conditions(init)

		# Result vector to pass to C library:
		result = np.zeros(d + 1)	

		# Run solver:
		resultArg = result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
		lib.Shotgun_run_lasso(self.obj, resultArg, len(result))

		# Form results:
		w = result[0:-1]
		offset = result[-1]
		residuals = A * np.mat(w).T + offset - np.mat(self.y).T
		obj = 0.5*np.linalg.norm(residuals, ord=2)**2 + self.lam*np.linalg.norm(w, ord=1)

		sol = lambda:0
		sol.w = w
		sol.offset = offset
		sol.residuals = residuals
		sol.obj = obj
		return sol
		

	def solve_lasso(self, A, y, lam, init=None):
		"""Solves lasso problem"""
		if (init and init[0] == None):
			init = None

		self.attach_A(A)
		self.load_y(y)
		self.set_lambda(lam)
		return self.run_lasso(init)







	### LOG REG CODE NOT UP TO DATE###
	def solve_logreg(self, A, y, lam, init=None):
		if (init and init[0] == None):
			init = None

		self.set_A(A)
		self.load_A(A)
		self.set_y(y)
		self.set_lambda(lam)
		return self.run_logreg(init)

	def run_logreg(self, init):
		# Assumes y and lambda are already loaded
		self.load_A(self.A)	
		d = self.A.shape[1]

		if (init):
			self.set_initial_conditions(init)

		# Result vector:
		result = np.zeros(d + 1)	

		# Run solver:
		resultArg = result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
		lib.Shotgun_run_logreg(self.obj, resultArg, len(result))

		w = result[0:-1]
		offset = result[-1]
		#residuals = A * np.mat(w).T + offset - np.mat(self.y).T
		#obj = 0.5*np.linalg.norm(residuals, ord=2)**2 + self.lam*np.linalg.norm(w, ord=1)

		sol = lambda:0
		sol.w = w
		sol.offset = offset
		#sol.residuals = residuals
		#sol.obj = obj
		return sol
