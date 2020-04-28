from ShootingMethods import MultiShootingTemplate
import sympy
from sympy import *
from sympy.printing.ccode import C99CodePrinter
import numpy as np
from cvxopt import matrix, solvers

class MPC:
	
	def __init__(self, model, N, T, Q_0, R_0, eq_constraint, ineq_constraint, term_ineq_constraint, term_eq_constraint, code_gen_file_name):
		
		np.set_printoptions(threshold=np.nan)
		init_printing()
		self.printer = C99CodePrinter()
		
		self.gen_code = "import numpy as np\nimport math\n"
		
		self.model = model
		
		self.N = N
		self.T = T
		self.dt = T/N
		
		self.lambda_forward = self.model.getForwardIntegrationLambda()
		self.forward_integration = self.model.forward_integration
		
		x0 = MatrixSymbol('x0', model.NX, 1)
		u0 = MatrixSymbol('u0', model.NU, 1)
		params = MatrixSymbol('params', model.NP, 1)
		dt = Symbol('dt')
		
		self.x0 = x0
		self.params = params
		
		for i in range(0, model.NX):
			self.forward_integration = self.forward_integration.subs(self.model.state[i], x0[i])
		for i in range(0, model.NU):
			self.forward_integration = self.forward_integration.subs(self.model.control[i], u0[i])
		for i in range(0, model.NP):
			self.forward_integration = self.forward_integration.subs(self.model.params[i], param[i])
		self.forward_integration = self.forward_integration.subs(self.model.dt, dt)
		
		self.Q_0 = Q_0
		self.R_0 = R_0

		self.term_eq_constraint = term_eq_constraint

		self.ineq_constraint = ineq_constraint
		self.term_ineq_constraint = term_ineq_constraint

		self.sym_state_vector = MatrixSymbol('x', model.NX, N+1)
		self.sym_control_vector = MatrixSymbol('u', model.NU, N)

		self.sym_X = Matrix([])
		for i in range(0, self.N+1):
			state_vec = sympy.zeros(self.model.NX, 1)
			for j in range(0, self.model.NX):
				state_vec[j, 0] = self.sym_state_vector[j,i]
			self.sym_X = self.sym_X.col_join(state_vec)
			
			if(i == self.N):
				continue
			control_vec = sympy.zeros(self.model.NU, 1)
			for j in range(0, self.model.NU):
				control_vec[j, 0] = self.sym_control_vector[j,i]
			self.sym_X = self.sym_X.col_join(control_vec)
		
		self.gen_code += "def forward_integration(x0, u0, params, dt):\n\tx_next = "
		self.gen_code += sympy.printing.lambdarepr.lambdarepr(self.forward_integration)
		self.gen_code = self.gen_code.replace("ImmutableDenseMatrix", "np.array")
		self.gen_code += "\n\treturn x_next\n"
		
		self.sym_state_ref_vector = MatrixSymbol('x_ref', model.NX, N+1)
		self.sym_control_ref_vector = MatrixSymbol('u_ref', model.NU, N)

		self.P = np.zeros((N*len(Q_0)+len(R_0), N*len(Q_0)+len(R_0)))

		for i in range(0, N):
			self.P[i*len(Q_0):(i+1)*len(Q_0), i*len(Q_0):(i+1)*len(Q_0)] = Q_0*self.dt
		self.P[N*len(Q_0):, N*len(Q_0):] = self.R_0

		self.P = self.P
		
		self.gen_code+= "def P_mat(x, u, params):\n\tP = np.array("
		self.gen_code+= np.array2string(np.array(self.P), separator=',')
		self.gen_code+= ")\n\treturn P\n"
		
		self.sym_X_ref = Matrix([])
		for i in range(0, self.N+1):
			state_ref_vec = sympy.zeros(self.model.NX, 1)
			for j in range(0, self.model.NX):
				state_ref_vec[j, 0] = self.sym_state_ref_vector[j,i]
			self.sym_X_ref = self.sym_X_ref.col_join(state_ref_vec)
			
			if(i == self.N):
				continue
			control_ref_vec = sympy.zeros(self.model.NU, 1)
			for j in range(0, self.model.NU):
				control_ref_vec[j, 0] = self.sym_control_ref_vector[j,i]
			self.sym_X_ref = self.sym_X_ref.col_join(control_ref_vec)
			
		self.q = -2*self.P.T*self.sym_X_ref
		self.gen_code += "def q_mat(x_ref, u_ref):\n\tq = "
		self.gen_code += sympy.printing.lambdarepr.lambdarepr(self.q)
		self.gen_code = self.gen_code.replace("ImmutableDenseMatrix", "np.array")
		self.gen_code += "\n\treturn q\n"
		self.q = lambdify([self.sym_X_ref.T], np.array(np.squeeze(self.q)), 'numpy')
		
		self.code_gen_file_name = code_gen_file_name
		
	def initialiseEqualityConstraints(self):
		# Multiple-shooting constraint
		self.h = sympy.zeros(self.model.NX, 1)
		for j in range(0, self.model.NX):
			self.h[j, 0] = self.sym_state_vector[j,0]-self.x0[j,0]
		self.h = self.h.col_join(MultiShootingTemplate.getEqualityConstraintMatrices(self.N, self.model, self.sym_state_vector, self.sym_control_vector, self.params))
		# Terminal eq constraint
		if(self.term_eq_constraint != None):
			self.h.col_join(self.term_eq_constraint(self.sym_state_vector[0:,-1], self.params))
		# discretisation time
		self.h = self.h.subs(self.model.dt, self.dt)

	def initialiseInequalityConstraints(self):
		
		self.g = Matrix([])
		# Terminal eq constraint
		if(self.ineq_constraint != None):
			self.g = self.g.col_join(self.ineq_constraint(self.sym_state_vector[0:,0:-1], self.sym_control_vector[0:,0:-1] , self.params))
		if(self.term_ineq_constraint != None):
			self.g = self.g.col_join(self.term_ineq_constraint(self.sym_state_vector[0:,-1], self.sym_control_vector[0:,-1], self.params))

	def lineariseConstraints(self):
		# linearisation
		self.A = self.h.jacobian(self.sym_X)
		self.b = -(self.h-self.A*self.sym_X)
		# printing to file
		self.gen_code += "def A_mat(x0, x, u, params):\n\tA = "
		self.gen_code += sympy.printing.lambdarepr.lambdarepr(self.A)
		self.gen_code = self.gen_code.replace("ImmutableDenseMatrix", "np.array")
		self.gen_code += "\n\treturn A\n"
		self.gen_code += "def b_mat(x0, x, u, params):\n\tb = "
		self.gen_code += sympy.printing.lambdarepr.lambdarepr(self.b)
		self.gen_code = self.gen_code.replace("ImmutableDenseMatrix", "np.array")
		self.gen_code += "\n\treturn b\n"
	
		# lambdifying
		self.A = lambdify([self.sym_X.T], np.squeeze(self.A.transpose()))
		self.b = lambdify([self.sym_X.T], np.squeeze(self.b.transpose()))

		if(self.ineq_constraint == None and self.term_ineq_constraint == None):
			self.gen_code += "noineq = True\n"
			return

		# Inequality constraints
		self.g = self.g.subs(self.model.dt, self.dt)

		self.G = self.g.jacobian(self.sym_X)
		self.h = -(self.g-self.G*self.sym_X)
		
		self.gen_code += "noineq = False\n"
		self.gen_code+= "def G_mat(x, u, params):\n\tG = np.array("
		self.gen_code+= np.array2string(np.array(self.G), separator=',')
		self.gen_code+= ")\n\treturn G\n"
		self.gen_code+= "def h_mat(x, u, params):\n\th = np.array("
		self.gen_code+= np.array2string(np.array(self.h), separator=',')
		self.gen_code+= ")\n\treturn h\n"
		
		text_file = open(self.code_gen_file_name+".py", "w")
		text_file.write(self.gen_code)
		text_file.close()

		# lambdifying
		self.G = lambdify([self.sym_X.T], np.squeeze(self.G))
		self.h = lambdify([self.sym_X.T], np.squeeze(self.h))
