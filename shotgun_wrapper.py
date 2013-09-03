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
class ShotgunLasso(object):

	def __init__(self):
		self.obj = lib.Shotgun_new()
		
	def set_A(self, A):
		# Sets Nxd data matrix
		if (A.ndim != 2):
			raise Exception("A must be 2d")
		if (np.iscomplex(A).any()):
			raise Exception("Sorry, imaginary values are not supported")

		self.d = A.shape[1]
		self.A = A

		(N, d) = A.shape

		if (sparse.issparse(A)):
			if (not sparse.isspmatrix_csc(A)):
				A = A.tocsc()
			indicesArg = A.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
			dataArg = A.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
			nnzArg = ctypes.c_int(A.nnz)
			lib.Shotgun_set_A_sparse(self.obj, dataArg, indicesArg, nnzArg, ctypes.c_int(N), ctypes.c_int(d))
		else:
			matrixArg = A.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
			lib.Shotgun_set_A(self.obj, matrixArg, ctypes.c_int(N), ctypes.c_int(d))

	def set_y(self, y):
		# Sets Nx1 labels matrix
		if (np.iscomplex(y).any()):
			raise Exception("Sorry, imaginary values are not supported")

		self.y = y
		arrayArg = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
		lib.Shotgun_set_y(self.obj, arrayArg, len(y)) 

	def set_lambda(self, value):
		# Sets regularization parameter lambda
		if (not np.isreal(value) or value < 0):
			raise Exception("Lambda must be a nonnegative real value")
		self.lam = value
		lib.Shotgun_set_lambda(self.obj, ctypes.c_double(value))

	def run(self):
		# Runs shotgun-lasso
		# Returns a solution object with several attributes
		if (self.y.shape[0] != self.A.shape[0]):
			raise Exception("A and y must have same number of training examples")	

		result = np.zeros(self.d + 1)	
		resultArg = result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
		lib.Shotgun_run(self.obj, resultArg, len(result))

		w = result[0:-1]
		offset = result[-1]
		residuals = self.A * np.mat(w).T + offset - np.mat(self.y).T
		obj = 0.5*np.linalg.norm(residuals, ord=2)**2 + self.lam*np.linalg.norm(w, ord=1)

		sol = lambda:0
		sol.w = w
		sol.offset = offset
		sol.residuals = residuals
		sol.obj = obj
		return sol

	def solve_lasso(self, A, y, _lambda):
		self.set_A(A)
		self.set_y(y)
		self.set_lambda(_lambda)
		return self.run()
