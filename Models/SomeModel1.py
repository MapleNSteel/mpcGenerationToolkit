from sympy import *
from Models.Model import Model
	
class SomeModel1(Model):

	x1, x2, u, t = symbols('x1, x2, u, t')
	dt = symbols('dt')

	NX = 2; NU = 1; NP = 0;

	params = Matrix([])
	state = Matrix([x1, x2])
	control = Matrix([u])

	def __init__(self, integration_method):
		self.integration_method = integration_method

	def f(self, state, control, params, t):

		x1, x2 = state[0], state[1];
		u = control[0];

		fx1 = 0;
		fx2 = u;
	
		f = Matrix([fx1, fx2])

		return f

	def getForwardIntegration(self, state, control, params, t):
		return self.integration_method(self.f)
