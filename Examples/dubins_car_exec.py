import time
import numpy as np
from Code_Gen import code_gen_dubins_car as code_gen
from Solver import Solver

# Initial Conditions
x1_0 = 0.0; x2_0 = 0.0; x3_0 = np.pi/2; x4_0 = 5.0; x5_0 = 0.;
N = code_gen.N; T = code_gen.T;
NX = code_gen.NX; NU = code_gen.NU; NP = code_gen.NP
num_iter = 10;
# initial values
X_0 = np.zeros((NX, N+1));
X_0[:,0] = np.array([x1_0, x2_0, x3_0, x4_0, x5_0]);
U_0 = np.zeros((NU, N));

params = np.zeros((0, N))

X_ref = np.zeros((NX, N+1));
U_ref = np.zeros((NU, N));

X_ref = np.tile(np.array([0., 5., np.pi/2, 5., 0.]), (NX, N+1))

X_f = X_0
U_f = U_0

soln = None
elapsed_time = 0.
for i in range(0, num_iter):
	start_time = time.perf_counter()
	soln = Solver.getSolution(code_gen, X_f, U_f, X_ref, U_ref, params)
	elapsed_time += time.perf_counter()-start_time
	for j in range(0, len(soln['x'])):
		if (j%(NX+NU) < NX):
			pass
		else:
			U_f[j%(NX+NU)-NX, int(j/(NX+NU))] = soln['x'][j]
	for j in range(0, N):
		X_f[:,j+1] = np.transpose(code_gen.forward_integration(np.reshape(X_f[:,j:j+1], (NX, 1)), np.reshape(U_f[:,j:j+1], (NU, 1)), np.reshape(params[:,j:j+1], (NP, 1)), T/N))

U_f = np.squeeze(U_f)

#print(np.shape(X_f))
#print(np.shape(U_f))
print(X_f)
print(U_f)
print(elapsed_time)
#print(soln['x'])

# plt.show()
