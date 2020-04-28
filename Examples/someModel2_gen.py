from IntegrationMethods.symbolicRK4 import *
from Models.SomeModel2 import SomeModel2
import numpy as np
from MPC.MPC import *
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)

N = 10; T = 1.;

Q = np.eye(SomeModel2.NX+SomeModel2.NU)
R = np.eye(SomeModel2.NX)*5

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

SomeModel2 = SomeModel2(RK4)

code_gen_file_name = "Code_Gen/code_gen_inverted_pendulum_mobile"

#Initialise MPC
mpc = MPC(SomeModel2, N, T, Q, R, eq_constraint, ineq_constraint, term_ineq_constraint, term_eq_constraint, code_gen_file_name)
mpc.initialiseEqualityConstraints()
mpc.initialiseInequalityConstraints()
mpc.lineariseConstraints()
