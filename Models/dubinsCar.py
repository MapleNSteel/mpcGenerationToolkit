from sympy import *
from RK4derivative import *
init_printing(use_unicode=True)

def dubinscar(state, control, t):
	# state = [x, y, psi, v, k], control = [a, u, t]
	x, y, psi, v, k = state[0], state[1], state[2], state[3], state[4];
	a, u = control[0], control[1]

	fx = v*cos(psi)
	fy = v*sin(psi)
	fpsi = v*k
	fv = a
	fk = u

	return Matrix([fx, fy, fpsi, fv, fk])

x, y, psi, v, k, a, u, t = symbols('x, y, psi, v, k, a, u, t')
dt = symbols('dt')

state = Matrix([x, y, psi, v, k])
control = Matrix([a, u])

next_state, jacobian_RK4_x, jacobian_RK4_u = RK4(someModel, state, control, t, dt, trigsimp)

pprint(next_state)
pprint(jacobian_RK4_x)
pprint(jacobian_RK4_u)
