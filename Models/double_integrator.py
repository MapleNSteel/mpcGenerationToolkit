from sympy import *
from Models.Model import Model
import numpy as np
	
class DoubleIntegrator(Model):

	x, v, a, t = symbols('x, v, a, t')
	dt = symbols('dt')

	NX = 2; NU = 1; NP = 0;

	params = Matrix([])
	state = Matrix([x, v])
	control = Matrix([a])

	def __init__(self, integration_method):
		self.integration_method = integration_method
		self.forward_integration = self.integration_method(self)
		self.forward_integration_lambda = lambdify([self.state, self.control, self.params, self.dt], np.squeeze(self.forward_integration))

	def f(self, state, control, params, t):

		x, v = state[0], state[1];
		a = control[0];

		fx1 = v;
		fx2 = a;
		
		f = Matrix([fx1, fx2])

		return f

	def getForwardIntegration(self):
		return self.forward_integration
		
	def getForwardIntegrationLambda(self):
		return self.forward_integration_lambda
