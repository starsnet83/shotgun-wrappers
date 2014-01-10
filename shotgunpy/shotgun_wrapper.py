import numpy as np
import scipy.sparse as sparse
import ctypes
import os


# Load Shotgun library:
dir = os.path.dirname(__file__)
libraryPath = os.path.join(dir, '../shotgun/shotgun_api.so')
lib = np.ctypeslib.load_library(libraryPath, dir)


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
lib.Shotgun_set_maxIter.restype = None
lib.Shotgun_set_maxIter.argtypes = [ctypes.c_void_p, ctypes.c_int]
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
		lib.Shotgun_set_threshold(self.obj, ctypes.c_double(value))

	def set_maxIter(self, value):
		"""Max iterations Shotgun will run"""
		lib.Shotgun_set_maxIter(self.obj, ctypes.c_int(value))

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

	def solve_lasso(self, A, y, lam, init=None):
		"""Helper method that calls the C library to solve lasso.
		On the argument A.  Assumes y and lambda are already loaded."""

		# Set-up:
		self.load_A(A)
		self.load_y(y)
		self.set_lambda(lam)
		if (init):
			self.set_initial_conditions(init)

		# Result vector to pass to C library:
		d = A.shape[1]
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
		sol.residuals = np.array(residuals).flatten()
		sol.obj = obj
		return sol

	def solve_logreg(self, A, y, lam, init=None):
		"""Helper method that calls the C library to solve logreg.
		On the argument A.  Assumes y and lambda are already loaded.
        Returned residuals = - y .* (A w + offset)."""

        # Set-up:
		self.load_A(A)
		self.load_y(y)
		self.set_lambda(lam)
		if (init):
			self.set_initial_conditions(init)

		# Result vector to pass to C library:
		d = A.shape[1]
		result = np.zeros(d + 1)	

		# Run solver:
		resultArg = result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
		lib.Shotgun_run_logreg(self.obj, resultArg, len(result))

		# Form results:
		w = result[0:-1]
		offset = result[-1]
		residuals = - np.multiply(A * np.mat(w).T + offset, np.mat(self.y).T)
        obj = self.lam*np.linalg.norm(w, ord=1)
        for i in range(len(residuals)):
            if (residuals[i] > (-10) and residuals[i] < 10):
                obj += np.sum(np.log(1.0 + np.exp(residuals[i])))
            elif (residuals[i] <= (-10)):
                obj += 0.0
            else:
                obj += residuals[i]
		sol = lambda:0
		sol.w = w
		sol.offset = offset
		sol.residuals = residuals
		sol.obj = obj
		return sol

