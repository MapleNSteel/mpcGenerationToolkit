from IntegrationMethods.symbolicRK4 import *
from Models.double_integrator import DoubleIntegrator
import numpy as np
from MPC.MPC import *

N = 10; T = 1.;

Q = np.eye(DoubleIntegrator.NX+DoubleIntegrator.NU)
R = 5np.eye(DoubleIntegrator.NX)

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

term_ineq_constraint = ineq_constraint
term_eq_constraint = None

DoubleIntegrator = DoubleIntegrator(RK4)

code_gen_file_name = "Code_Gen/code_gen_double_integrator"
matrix_form = "2d"
ineq_form = "l_ineq"

#Initialise MPC
mpc = MPC(DoubleIntegrator, N, T, Q, R, eq_constraint, term_eq_constraint, ineq_constraint, term_ineq_constraint, code_gen_file_name, matrix_form, ineq_form)
mpc.lineariseObjective()
mpc.initialiseEqualityConstraints()
mpc.initialiseInequalityConstraints()
mpc.lineariseConstraints()
mpc.writeGenCode()
