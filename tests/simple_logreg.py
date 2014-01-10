import numpy as np
import shotgunpy

N = 5
d = 10
k = 2
A = np.random.randn(N, d)
w = np.zeros(d)
w[0:k] = 1
noise = np.random.randn(N)
y = 2. * (np.dot(A, w) + noise > 0) - 1


s = shotgunpy.ShotgunSolver()
s.set_use_offset(False)

lammax = max(abs(np.dot(A.T, y)))/2.
lam = 0.8*lammax
sol = s.solve_logreg(A, y, lam)

print sol.w
