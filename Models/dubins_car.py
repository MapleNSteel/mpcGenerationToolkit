from sympy import *
from Models.Model import Model
import numpy as np

class DubinsCar(Model):

	x, y, psi, v, k, a, u, t = symbols('x, y, psi, v, k, a, u, t')
	dt = symbols('dt')
	
	NX = 5; NU = 2; NP = 0;
	
	params = Matrix([])
	state = Matrix([x, y, psi, v, k])
	control = Matrix([a, u])

	def __init__(self, integration_method):
		
		self.integration_method = integration_method
		self.forward_integration = self.integration_method(self)
		self.forward_integration_lambda = lambdify([self.state, self.control, self.params, self.dt], np.squeeze(self.forward_integration))
		
	def f(self, state, control, params, t):
	
		x, y, psi, v, k = state[0], state[1], state[2], state[3], state[4];
		a, u = control[0], control[1]

		fx = v*cos(psi)
		fy = v*sin(psi)
		fpsi = v*k
		fv = a
		fk = u
		
		f = Matrix([fx, fy, fpsi, fv, fk])

		return f
		
	def getForwardIntegration(self):
		return self.forward_integration
		
	def getForwardIntegrationLambda(self):
		return self.forward_integration_lambda
