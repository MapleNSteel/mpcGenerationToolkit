from IntegrationMethods.symbolicRK4 import *
from Models.double_integrator import DoubleIntegrator
import numpy as np
from MPC.MPC import *

N = 1; T = 0.1;

Q = np.eye(DoubleIntegrator.NX+DoubleIntegrator.NU)
R = 5*np.eye(DoubleIntegrator.NX)

eq_constraint = None
def ineq_constraint(state_vec, control_vec, params):
	g = Matrix([])
	
	for i in range(0, state_vec.shape[1]):
		g = g.col_join(Matrix([state_vec[0, i]-1.5]))
		g = g.col_join(Matrix([-state_vec[0, i]-1.5]))

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
