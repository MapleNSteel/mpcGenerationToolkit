import numpy as np
from sympy import *

def initialiseStatesAndControl(X):
	return

def getEqualityConstraintMatrices(N, model, sym_state_vector, sym_control_vector, params):

	h = []
	f = model.getForwardIntegration()

	for i in range(0, N):
		temp_f = f;
		for j in range(0, model.NX):
			temp_f = temp_f.subs(model.state[j], sym_state_vector[j,i])
			temp_f[j,0] = temp_f[j,0] - sym_state_vector[j,i+1]
		for j in range(0, model.NU):
			temp_f = temp_f.subs(model.control[j], sym_control_vector[j,i])
		for j in range(0, model.NP):
			temp_f = temp_f.subs(model.params[j], params[j,i])
		h.append(expand(temp_f))

	h = Matrix(h)
	return h
