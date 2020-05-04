from IntegrationMethods.symbolicRK4 import *
from Models.double_integrator import DoubleIntegrator
import numpy as np
from MPC.MPC import *

N = 10; T = 1.;

Q = 1e-3*np.eye(DoubleIntegrator.NX+DoubleIntegrator.NU)
Q[0, 0] = 1
R = np.eye(DoubleIntegrator.NX)
R[0, 0] = 5

eq_constraint = None
def ineq_constraint(state_vec, control_vec, params):
	g = Matrix([])
	
	for i in range(0, state_vec.shape[1]):
		g = g.col_join(Matrix([state_vec[1, i]-1.5]))
		g = g.col_join(Matrix([-state_vec[1, i]-1.5]))

	for i in range(0, control_vec.shape[1]):
		g = g.col_join(Matrix([control_vec[0, i]-1]))
		g = g.col_join(Matrix([-control_vec[0, i]-1]))
	return g

term_ineq_constraint = None
term_eq_constraint = None

DoubleIntegrator = DoubleIntegrator(RK4)

code_gen_file_name = "Code_Gen/code_gen_double_integrator"

#Initialise MPC
mpc = MPC(DoubleIntegrator, N, T, Q, R, eq_constraint, ineq_constraint, term_ineq_constraint, term_eq_constraint, code_gen_file_name)
mpc.initialiseEqualityConstraints()
mpc.initialiseInequalityConstraints()
mpc.lineariseConstraints()
