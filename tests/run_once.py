# Load a Matlab file, and run
import unittest
import numpy as np
import scipy.io
import sys
sys.path.append('..')
import shotgunpy

data = scipy.io.loadmat("/Users/jbradley/data/lasso/lasso2_1000_1000_1.mat")
y = np.array(data['y'], dtype=np.float)
A = data['A']
print 'Running Lasso on dataset with n=%d, d=%d\n' % np.shape(data['A'])

solver = shotgunpy.ShotgunSolver()
solver.set_use_offset(False)
solver.set_maxIter(10)
lam = .5
sol = solver.solve_lasso(A,y,lam)
sol.obj
print 'Final objective = %g\n' % (sol.obj)
#print 'Final solution = %s\n' % (str(sol.w))

class TestPythonWrapper(unittest.TestCase):

	def setUp(self):
		self.solver = shotgunpy.ShotgunSolver()
		self.solver.set_use_offset(False)

	def test_lasso_simple(self):
		A = np.array([[1, 1], [1, -1]], dtype=np.float)
		y = np.array([1, 0], dtype=np.float)
		lam = 0.5
		sol = self.solver.solve_lasso(A, y, lam)
		self.assertEqual(sol.w[0], 0.25)
		self.assertEqual(sol.w[1], 0.25)

	def test_lasso_offset(self):
		self.solver.set_use_offset(True)
		A = np.eye(3)
		y = np.array([1, 1, 1], dtype=np.float)
		sol = self.solver.solve_lasso(A, y, 0.5)
		self.assertEqual(sol.offset, 1.)

if __name__ == '__main__':
	unittest.main()
