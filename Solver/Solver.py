import time
import numpy as np
from cvxopt import matrix, solvers
from sympy import pprint
solvers.options['show_progress'] = False
solvers.options['maxiters'] = 1

def getSolution(code_gen, x_0, u_0, x_ref, u_ref, params):
	A = code_gen.A_mat(x_0[:,0:1], x_0, u_0, params)
	b = code_gen.b_mat(x_0[:,0:1], x_0, u_0, params)
	P = code_gen.P_mat(x_0, u_0, params)
	q = code_gen.q_mat(x_0, u_0, x_ref, u_ref)
	#pprint(A)
	#pprint(b)
	if(code_gen.noineq != True):
		G = code_gen.G_mat(x_0, u_0, params)
		h = code_gen.h_mat(x_0, u_0, params)
		
		return solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))
	else:
		return solvers.qp(matrix(P), matrix(q), None, None, matrix(A), matrix(b))
