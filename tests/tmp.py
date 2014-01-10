# Load a Matlab file, and run
import unittest
import numpy as np
import scipy.io
import sys
sys.path.append('..')
import shotgunpy

data = scipy.io.loadmat("/Users/jbradley/data/lasso/arcene.mat")
y = np.array(data['y'], dtype=np.float)
A = data['A']

solver = shotgunpy.ShotgunSolver()
solver.set_use_offset(False)
solver.set_maxIter(10)
lam = .5
sol = solver.solve_lasso(A,y,lam)
sol.obj

