# Load a Matlab file, and run
import unittest
import numpy as np
import scipy.io
import sys
sys.path.append('..')
import shotgunpy

data = scipy.io.loadmat("/Users/jbradley/data/lasso/a9a.mat")
y = np.array(data['y'], dtype=np.float)
A = data['A']

solver = shotgunpy.ShotgunSolver()
solver.set_use_offset(False)
solver.set_maxIter(10000)
lam = .1
sol = solver.solve_logreg(A,y,lam)
print str(np.linalg.norm(sol.w,1))
print str(sol.obj)

