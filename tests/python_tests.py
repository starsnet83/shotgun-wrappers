import unittest
import numpy as np
import shotgunpy

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
