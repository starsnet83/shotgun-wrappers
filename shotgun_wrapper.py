import numpy as np
import scipy.sparse as sparse
import ctypes
import os

# Load Shotgun library:
dir = os.path.dirname(__file__)
libraryPath = os.path.join(dir, 'shotgun_api.so')
lib = ctypes.cdll.LoadLibrary(libraryPath)
lib.Shotgun_run.restype = ctypes.POINTER(ctypes.c_double)

# Define Shotgun interface:
class ShotgunSolver(object):

	def __init__(self):
		self.obj = lib.Shotgun_new()
		self.set_num_threads(1)
		
	def set_A(self, A):
		# Sets Nxd data matrix
		if (A.ndim != 2):
			raise Exception("A must be 2d")
		if (np.iscomplex(A).any()):
			raise Exception("Sorry, imaginary values are not supported")
		self.A = A
		self.d = A.shape[1]

	def load_A(self, A):
		(N, d) = A.shape

		if (sparse.issparse(A)):
			if (not sparse.isspmatrix_csc(A)):
				A = A.tocsc()
			indicesArg = A.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
			dataArg = A.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
			indptrArg = A.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
			nnzArg = ctypes.c_int(A.nnz)
			lib.Shotgun_set_A_sparse(self.obj, dataArg, indicesArg, indptrArg, nnzArg, ctypes.c_int(N), ctypes.c_int(d))
		else:
			matrixArg = A.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
			lib.Shotgun_set_A(self.obj, matrixArg, ctypes.c_int(N), ctypes.c_int(d), A.ctypes.strides)

	def set_y(self, y):
		# Sets Nx1 labels matrix
		if (np.iscomplex(y).any()):
			raise Exception("Sorry, imaginary values are not supported")

		self.y = y.flatten()
		arrayArg = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
		lib.Shotgun_set_y(self.obj, arrayArg, ctypes.c_int(len(y))) 

	def set_lambda(self, value):
		# Sets regularization parameter lambda
		if (not np.isreal(value) or value < 0):
			raise Exception("Lambda must be a nonnegative real value")
		self.lam = value
		lib.Shotgun_set_lambda(self.obj, ctypes.c_double(value))

	def set_tolerance(self, value):
		lib.Shotgun_set_threshold(self.obj, ctypes.c_double(value))

	def set_use_offset(self, value):
		lib.Shotgun_set_use_offset(self.obj, ctypes.c_int(value))

	def set_num_threads(self, value):
		if (type(value) != int):
			raise Exception("Number of threads should be an integer")
		self.numThreads = value
		lib.Shotgun_set_num_threads(self.obj, ctypes.c_int(value))

	def set_initial_conditions(self, (w, offset)):
		wArg = w.ctypes.data_as(ctypes.POINTER(ctypes.c_double))	
		offsetArg = ctypes.c_double(offset)
		lib.Shotgun_set_initial_conditions(self.obj, wArg, offsetArg)

	def run(self, initialConditions):
		# Runs shotgun-lasso
		# Returns a solution object with several attributes

		print "Running..."
		if (self.y.shape[0] != self.A.shape[0]):
			raise Exception("A and y must have same number of training examples")	

		if (initialConditions):
			(w, offset) = initialConditions
			residuals = self.A * np.mat(w).T + offset - np.mat(self.y).T
			residuals = np.array(residuals).flatten()
		else:
			w = np.zeros(self.d) 
			offset = self.y.mean()
			residuals = self.y - offset

		cutoff = 0.95 * self.lam

		obj = float('inf')
		print "Main loop"
		while True:
			subGrad = np.array(self.A.T * np.mat(residuals).T).flatten()
			currentIndices = np.where(abs(subGrad) > cutoff)[0]
			currentA = self.A[:, currentIndices]
			wInit = w[currentIndices]
			offsetInit = offset
			sol = self.solve_lasso_subproblem(currentA, (wInit, offsetInit))
			w = np.zeros(self.d)
			w[currentIndices] = sol.w
			if (abs(sol.obj - obj) < 1e-4):
				sol.w = w
				break
			else:
				obj = sol.obj
				residuals = np.array(sol.residuals).flatten()
				offset = sol.offset

		return sol

	def solve_lasso_subproblem(self, A, init=None):
		# Assumes y and lambda are already loaded
		self.load_A(A)	
		d = A.shape[1]

		if (init):
			self.set_initial_conditions(init)

		# Result vector:
		result = np.zeros(d + 1)	

		# Complicated mess to ensure convergence:
		initialNumThreads = self.numThreads
		while True:
			# Run solver:
			resultArg = result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
			lib.Shotgun_run(self.obj, resultArg, len(result))

			# Try to make sure things converged (bit of a hack for now):
			offset = result[-1]
			if (np.isnan(offset)):
				# Not converged so reduce number of threads...
				newNumThreads = self.numThreads/2
				self.set_num_threads(newNumThreads)
				print "Warning: shotgun diverged, reducing number of parallel updates to " + str(newNumThreads)
			else:
				self.set_num_threads(initialNumThreads)
				break

		w = result[0:-1]
		residuals = A * np.mat(w).T + offset - np.mat(self.y).T
		obj = 0.5*np.linalg.norm(residuals, ord=2)**2 + self.lam*np.linalg.norm(w, ord=1)

		sol = lambda:0
		sol.w = w
		sol.offset = offset
		sol.residuals = residuals
		sol.obj = obj
		return sol
		

	def solve_lasso(self, A, y, lam, init=None):
		self.set_A(A)
		self.set_y(y)
		self.set_lambda(lam)
		return self.run(init)
