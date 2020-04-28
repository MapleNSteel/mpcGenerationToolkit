import numpy as np
from Code_Gen import code_gen_inverted_pendulum_mobile as code_gen
from Solver import Solver

# Initial Conditions
x1_0 = 0.1; x2_0 = 0.1; x3_0 = np.pi/2; x4_0 = 0.1; N = 10; T = 1.; num_iter = 10;
# initial values
X_0 = np.zeros((4, N+1));
X_0[:,0] = np.array([x1_0, x2_0, x3_0, x4_0]);
U_0 = np.zeros((1, N));

params = np.zeros((0, N))

X_ref = np.zeros((4, N+1));
U_ref = np.zeros((1, N));

X_f = X_0
U_f = U_0

soln = None

for i in range(0, num_iter):
	soln = Solver.getSolution(code_gen, X_f, U_f, X_ref, U_ref, params)	
	for j in range(0, len(soln['x'])):
		if (j%(4+1) < 4):
			pass
		else:
			U_f[j%(4+1)-4, int(j/(4+1))] = soln['x'][j]
	for j in range(0, N):
		X_f[:,j+1] = np.transpose(code_gen.forward_integration(np.reshape(X_f[:,j:j+1], (4, 1)), np.reshape(U_f[:,j:j+1], (1, 1)), np.reshape(params[:,j:j+1], (0, 1)), T/N))

U_f = np.squeeze(U_f)

#print(np.shape(X_f))
#print(np.shape(U_f))
print(X_f)
print(U_f)
#print(soln['x'])

# plt.show()
