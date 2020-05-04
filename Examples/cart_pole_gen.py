from IntegrationMethods.symbolicRK4 import *
from Models.cart_pole import CartPole
import numpy as np
from MPC.MPC import *

N = 2; T = 1;

Q = 1e-1*np.eye(CartPole.NX+CartPole.NU)
Q[0, 0] = 2
R = np.eye(CartPole.NX)
R[0, 0] = 20

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

CartPole = CartPole(RK4)

code_gen_file_name = "Code_Gen/code_gen_cart_pole"

#Initialise MPC
mpc = MPC(CartPole, N, T, Q, R, eq_constraint, ineq_constraint, term_ineq_constraint, term_eq_constraint, code_gen_file_name)
mpc.initialiseEqualityConstraints()
mpc.initialiseInequalityConstraints()
mpc.lineariseConstraints()
