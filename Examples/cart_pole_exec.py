import time
import numpy as np
from Code_Gen import code_gen_cart_pole as code_gen
from Solver import Solver
import matplotlib.pyplot as plt

# Initial Conditions
x1_0 = 0.0; x2_0 = 0.0; x3_0 = 0.; x4_0 = 0.0; 
N = code_gen.N; T = code_gen.T;
NX = code_gen.NX; NU = code_gen.NU; NP = code_gen.NP
num_iter = 5;
num_steps = 100;
# initial values
X_0 = np.zeros((NX, N+1));
X_0[:,0] = np.array([x1_0, x2_0, x3_0, x4_0]);
U_0 = np.zeros((NU, N));

params = np.zeros((0, N))

X_ref = np.zeros((NX, N+1));
U_ref = np.zeros((NU, N));

X_ref = np.tile(np.array([[0.1], [0.0], [0.0], [0.0]]), (1, N+1))

X_f = X_0
U_f = U_0

soln = None
elapsed_time = 0.

pos_vec = np.zeros(num_steps)

for j in range(0, N):
		X_f[:,j+1] = np.transpose(code_gen.forward_integration(np.reshape(X_f[:,j:j+1], (NX, 1)), np.reshape(U_f[:,j:j+1], (NU, 1)), np.reshape(params[:,j:j+1], (NP, 1)), T/N))

for i in range(0, num_steps):
	for j in range(0, num_iter):
		start_time = time.perf_counter()
		soln = Solver.getSolution(code_gen, X_f, U_f, X_ref, U_ref, params)
		elapsed_time += time.perf_counter()-start_time
		
		for k in range(0, len(soln['x'])):
			if (k%(NX+NU) < NX):
				pass
			else:
				U_f[k%(NX+NU)-NX, int(k/(NX+NU))] = soln['x'][k]
		for k in range(0, N):
			X_f[:,k+1] = np.transpose(code_gen.forward_integration(np.reshape(X_f[:,k:k+1], (NX, 1)), np.reshape(U_f[:,k:k+1], (NU, 1)), np.reshape(params[:,k:k+1], (NP, 1)), T/N))

	#print(np.shape(X_f))
	#print(np.shape(U_f))
	#print(X_f[0,0:1])
	#print(U_f[:,0:1])
	
	pos_vec[i] = X_f[0, 0:1]
	
	X_f[:,0] = np.transpose(code_gen.forward_integration(np.reshape(X_f[:,0:1], (NX, 1)), np.reshape(U_f[:,0:1], (NU, 1)), np.reshape(params[:,0:1], (NP, 1)), T/N))
	for k in range(0, N):
		X_f[:,k+1] = np.transpose(code_gen.forward_integration(np.reshape(X_f[:,k:k+1], (NX, 1)), np.reshape(U_f[:,k:k+1], (NU, 1)), np.reshape(params[:,k:k+1], (NP, 1)), T/N))

plt.plot(np.linspace(0, num_steps*(T/N), num_steps), pos_vec)
plt.show()

print(pos_vec)
print(elapsed_time)
