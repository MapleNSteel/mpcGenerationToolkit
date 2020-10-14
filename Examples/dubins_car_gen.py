from IntegrationMethods.symbolicRK4 import *
from Models.dubins_car import DubinsCar
import numpy as np
from MPC.MPC import *

N = 10; T = 1.;

Q = np.eye(DubinsCar.NX+DubinsCar.NU)
R = np.eye(DubinsCar.NX)*5

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

DubinsCar = DubinsCar(RK4)

code_gen_file_name = "Code_Gen/code_gen_dubins_car"
matrix_form = "2d"
ineq_form = "l_ineq"

#Initialise MPC
mpc = MPC(DubinsCar, N, T, Q, R, eq_constraint, term_eq_constraint, ineq_constraint, term_ineq_constraint, code_gen_file_name, matrix_form, ineq_form)
mpc.lineariseObjective()
mpc.initialiseEqualityConstraints()
mpc.initialiseInequalityConstraints()
mpc.lineariseConstraints()
mpc.writeGenCode()
