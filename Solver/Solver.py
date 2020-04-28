import numpy as np
from cvxopt import matrix, solvers

def getSolution(code_gen, x_0, u_0, x_ref, u_ref, params):
	A = np.array(code_gen.A_mat(x_0[:,0:1], x_0, u_0, params)).astype(float)
	print(np.shape(A))
	b = np.array(code_gen.b_mat(x_0[:,0:1], x_0, u_0, params)).astype(float)
	P = np.array(code_gen.P_mat(x_0, u_0, params))
	q = np.array(code_gen.q_mat(x_ref, u_ref)).astype(float)
	if(code_gen.noineq != True):
		G = np.array(code_gen.G_mat(x_0, u_0, params)).astype(float)
		h = np.array(code_gen.h_mat(x_0, u_0, params)).astype(float)
		
		return solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))
	else:
		return solvers.qp(matrix(P), matrix(q), None, None, matrix(A), matrix(b))
