from sympy import *

def RK4(model):
	
	f, state, control, params, t, dt = model.f, model.state, model.control, model.params, model.t, model.dt

	# RK4 step dt

	k1 = dt*f(state, control, params, t)
	k2 = dt*f(state+k1/2, control, params, t+dt/2)
	k3 = dt*f(state+k2/2, control, params, t+dt/2)
	k4 = dt*f(state+k3, control, params, t+dt)

	next_state = (state + (k1+2*k2+2*k3+k4)/6)

	return next_state
