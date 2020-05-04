from sympy import *
from Models.Model import Model
import numpy as np
	
class CartPole(Model):

	p, v, phi, omega, a, t = symbols('p, v, phi, omega, a, t')
	dt = symbols('dt')

	NX = 4; NU = 1; NP = 0;

	params = Matrix([])
	state = Matrix([p, v, phi, omega])
	control = Matrix([a])

	def __init__(self, integration_method):
		self.integration_method = integration_method
		self.forward_integration = self.integration_method(self)
		self.forward_integration_lambda = lambdify([self.state, self.control, self.params, self.dt], np.squeeze(self.forward_integration))

	def f(self, state, control, params, t):

		p, v, phi, omega = state[0], state[1], state[2], state[3];
		a = control[0];

		fx1 = v;
		fx2 = a;
		fx3 = omega;
		fx4 = -9.81*sin(phi) - a*cos(phi) - 0.2*omega;
		
		f = Matrix([fx1, fx2, fx3, fx4])

		return f

	def getForwardIntegration(self):
		return self.forward_integration
		
	def getForwardIntegrationLambda(self):
		return self.forward_integration_lambda
