from ShootingMethods import MultiShootingTemplate
import sympy
from sympy import *
from sympy.printing.ccode import C99CodePrinter
import numpy as np
from cvxopt import matrix, solvers
import sys
from pathlib import Path

class MPC:

        global lang_list, matrix_form_list, ineq_form_list
        
        lang_list = ["cpp", "py"]
        matrix_form_list = ["1drow", "1dcol", "2d"]
        ineq_form_list = ["l_ineq", "lr_ineq"]
        
        def __init__(self, model, N, T, Q_0, R_0, eq_constraint, term_eq_constraint, ineq_constraint, term_ineq_constraint, code_gen_file_name, code_gen_language, matrix_form, ineq_form):
        
                global lang_list, matrix_form_list, ineq_form_list
                
                assert code_gen_language in lang_list
                assert matrix_form in matrix_form_list
                assert ineq_form in ineq_form_list
                
                self.code_gen_language = code_gen_language
                self.matrix_form = matrix_form
                
                np.set_printoptions(threshold=sys.maxsize)
                init_printing()
                self.printer = C99CodePrinter()
                
                self.model = model
                
                self.N = N
                self.T = T
                self.dt = T/N
                
                self.writeParams()
                
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
                
                self.eq_constraint = eq_constraint
                self.term_eq_constraint = term_eq_constraint

                self.ineq_constraint = ineq_constraint
                self.term_ineq_constraint = term_ineq_constraint
                
                self.code_gen_file_name = code_gen_file_name

                self.sym_state_vector = MatrixSymbol('x', model.NX, N+1)
                self.sym_control_vector = MatrixSymbol('u', model.NU, N)
                
                self.sym_state_ref_vector = MatrixSymbol('x_ref', model.NX, N+1)
                self.sym_control_ref_vector = MatrixSymbol('u_ref', model.NU, N)

                self.sym_X = Matrix([])
                self.sym_X_ref = Matrix([])
                self.J = Matrix([0])
                for i in range(0, self.N+1):
                        state_vec = sympy.zeros(self.model.NX, 1)
                        state_ref_vec = sympy.zeros(self.model.NX, 1)
                        for j in range(0, self.model.NX):
                                state_vec[j, 0] = self.sym_state_vector[j,i]
                                state_ref_vec[j, 0] = self.sym_state_ref_vector[j,i]
                        temp = state_vec
                        temp_ref = state_ref_vec
                        self.sym_X = self.sym_X.col_join(state_vec)
                        self.sym_X_ref = self.sym_X_ref.col_join(state_ref_vec)
                        
                        if(i == self.N):
                                self.J+= (temp-temp_ref).T*R_0*(temp-temp_ref)
                                continue
                        control_vec = sympy.zeros(self.model.NU, 1)
                        control_ref_vec = sympy.zeros(self.model.NU, 1)
                        for j in range(0, self.model.NU):
                                control_vec[j, 0] = self.sym_control_vector[j,i]
                                control_ref_vec[j, 0] = self.sym_control_ref_vector[j,i]
                        temp = temp.col_join(control_vec)
                        temp_ref = temp_ref.col_join(control_ref_vec)
                        self.sym_X = self.sym_X.col_join(control_vec)
                        self.sym_X_ref = self.sym_X_ref.col_join(control_ref_vec)
                        
                        self.J+= (temp-temp_ref).T*Q_0*(temp-temp_ref)
                        
                self.J *= 0.5
                self.sym_X = ImmutableMatrix(self.sym_X)
                self.sym_X_ref = ImmutableMatrix(self.sym_X_ref)
                
                if(self.code_gen_language == "py"):
                        self.gen_code += "def forward_integration(x0, u0, params, dt):\n\tx_next = "
                        self.gen_code += sympy.printing.lambdarepr.lambdarepr(self.forward_integration)
                        self.gen_code = self.gen_code.replace("ImmutableDenseMatrix", "np.array")
                        self.gen_code += ".astype(float)\n\treturn x_next\n"
                
                elif(self.code_gen_language == "cpp"):
                        if(self.matrix_form == "2d"):
                                self.gen_code += "std::array<float64_t, NX> forward_integration(std::array<float64_t, NX> x0, std::array<float64_t, NU> u0, std::array<float64_t, NP> params, float64_t dt)"
                        elif(self.matrix_form == "1drow" or self.matrix_form == "1dcol"):
                                self.gen_code += "float64_t* forward_integration(float64_t* x0, float64_t* u0, float64_t* params, float64_t dt)"
                                       
                        self.gen_code += "\n{"
                        if(self.matrix_form == "2d"):
                                self.gen_code += "\n\tstd::array<float64_t, NX> x_next;\n"
                        elif(self.matrix_form == "1drow" or self.matrix_form == "1dcol"):
                                self.gen_code += "\n\tfloat64_t* x_next = new float64_t[NX];\n"
                                
                        for i in range(0, self.forward_integration.shape[0]):
                                var = str(self.forward_integration[i])
                                var = var.replace(", ", "][")
                                for j in range(0, self.model.NX):
                                        var = var.replace("x0["+str(j)+"][0]", "x0["+str(j)+"]")
                                        var = var.replace("u0["+str(j)+"][0]", "u0["+str(j)+"]")
                                self.gen_code += "\tx_next["+str(i)+"] = "+var+";\n"
                        self.gen_code += "\n\treturn x_next;"
                        self.gen_code += "\n}\n"
                        
        def writeParams(self):
                if(self.code_gen_language == "py"):
                        self.gen_code = "import numpy as np\nimport math\n"
                        self.gen_code+= "N ="
                        self.gen_code+= str(self.N)
                        self.gen_code+= "\nT="
                        self.gen_code+= str(self.T)
                        self.gen_code+= "\nNX="
                        self.gen_code+= str(self.model.NX)
                        self.gen_code+= "\nNU="
                        self.gen_code+= str(self.model.NU)
                        self.gen_code+= "\nNP="
                        self.gen_code+= str(self.model.NP)
                        self.gen_code+= "\n"

                elif(self.code_gen_language == "cpp"):
                        self.gen_code = "#include <cmath>\n"
                        self.gen_code+= "#include <array>\n"
                        self.gen_code+= "#include <cstdint>\n"
                        self.gen_code+= "typedef double float64_t;\n"
                        self.gen_code+= "extern const size_t N = "
                        self.gen_code+= str(self.N)
                        self.gen_code+= ";\nextern const float64_t T = "
                        self.gen_code+= str(self.T)
                        self.gen_code+= ";\nextern const size_t NX = "
                        self.gen_code+= str(self.model.NX)
                        self.gen_code+= ";\nextern const size_t NU = "
                        self.gen_code+= str(self.model.NU)
                        self.gen_code+= ";\nextern const size_t NP = "
                        self.gen_code+= str(self.model.NP)
                        self.gen_code+= ";\n"
                
        def lineariseObjective(self):
                
                self.q = self.J.jacobian(self.sym_X).T
                self.P = self.q.jacobian(self.sym_X)
                
                for i in range(0, (self.N+1)*self.model.NX + self.N*self.model.NU):
                                self.q = self.q.subs(self.sym_X[i], 0)
                                self.P = self.P.subs(self.sym_X[i], 0)
                                                
                if(self.code_gen_language == "py"):                
                        self.gen_code+= "def P_mat(x, u, params):\n\tP = np.array("
                        self.gen_code+= np.array2string(np.array(self.P), separator=',')
                        self.gen_code+= ").astype(float)\n\treturn P\n"
                        
                        self.gen_code += "def q_mat(x, u, x_ref, u_ref):\n\tq = "
                        self.gen_code += sympy.printing.lambdarepr.lambdarepr(self.q)
                        self.gen_code = self.gen_code.replace("ImmutableDenseMatrix", "np.array")
                        self.gen_code += ".astype(float)\n\treturn q\n"
                        self.q = lambdify([self.sym_X_ref.T], np.array(np.squeeze(self.q)), 'numpy')                        
                        
                elif(self.code_gen_language == "cpp"):
                        if(self.matrix_form == "2d"):
                                self.gen_code += "std::array<std::array<float64_t, (N+1)*NX + N*NU>, (N+1)*NX + N*NU> P_mat(std::array<std::array<float64_t, NX>, N+1> x, std::array<std::array<float64_t, NU>, N> u, std::array<std::array<float64_t, NX>, N+1> x_ref, std::array<std::array<float64_t, NU>, N> u_ref, std::array<std::array<float64_t, NP>, N> params)\n{\n"
                                self.gen_code += "\tstd::array<std::array<float64_t, (N+1)*NX + N*NU>, ((N+1)*NX + N*NU)>  P;\n"
                        elif(self.matrix_form == "1drow" or self.matrix_form == "1dcol"):
                                self.gen_code += "float64_t* P_mat(float64_t* x, float64_t* u, float64_t* x_ref, float64_t* u_ref, float64_t* params)\n{\n"
                                self.gen_code += "\tfloat64_t* P = new float64_t[((N+1)*NX + N*NU)*((N+1)*NX + N*NU)];\n"
                        
                        num_rows = (self.N+1)*self.model.NX + self.N*self.model.NU
                        num_cols = (self.N+1)*self.model.NX + self.N*self.model.NU
                        
                        for i in range(0, num_rows):
                                for j in range(0, num_cols):
                                        var = str(self.P[i, j])
                                        var = var.replace(", ", "][")
                                        if(self.matrix_form == "2d"):
                                                self.gen_code += "\tP["+str(i)+"]["+str(j)+"] = "+var+";\n"
                                        elif(self.matrix_form == "1drow"):
                                                self.gen_code += "\tP["+str(i*num_cols + j)+"] = "+var+";\n"
                                        elif(self.matrix_form == "1dcol"):
                                                self.gen_code += "\tP["+str(i + j*num_rows)+"] = "+var+";\n"                                
                        self.gen_code += "\treturn P;\n"
                        self.gen_code += "}\n"
                        
                        if(self.matrix_form == "2d"):
                                self.gen_code += "std::array<float64_t, (N+1)*NX + N*NU> q_mat(std::array<std::array<float64_t, NX>, N+1> x, std::array<std::array<float64_t, NU>, N> u, std::array<std::array<float64_t, NX>, N+1> x_ref, std::array<std::array<float64_t, NU>, N> u_ref, std::array<std::array<float64_t, NP>, N> params)\n{\n"
                                self.gen_code += "\tstd::array<float64_t, (N+1)*NX + N*NU> q;\n"
                        
                        elif(self.matrix_form == "1drow" or self.matrix_form == "1dcol"):
                                self.gen_code += "float64_t* q_mat(float64_t* x, float64_t* u, float64_t* x_ref, float64_t* u_ref, float64_t* params)\n{\n"
                                self.gen_code += "\tfloat64_t* q = new float64_t[(N+1)*NX + N*NU];\n"
                                
                        for i in range(0, (self.N+1)*self.model.NX + self.N*self.model.NU):
                                var = str(self.q[i])
                                var = var.replace(", ", "][")
                                self.gen_code += "\tq["+str(i)+"] = "+var+";\n"
                        self.gen_code += "\treturn q;\n"
                        self.gen_code += "}\n"
                
        def initialiseEqualityConstraints(self):
                # Multiple-shooting constraint
                self.h = sympy.zeros(self.model.NX, 1)
                for j in range(0, self.model.NX):
                        self.h[j, 0] = self.sym_state_vector[j,0]-self.x0[j,0]
                self.h = self.h.col_join(MultiShootingTemplate.getEqualityConstraintMatrices(self.N, self.model, self.sym_state_vector, self.sym_control_vector, self.params))
                # Eq constraint
                if(self.term_eq_constraint != None):
                        self.h.col_join(self.eq_constraint(self.sym_state_vector[0:,-1], self.params))
                # Terminal eq constraint
                if(self.term_eq_constraint != None):
                        self.h.col_join(self.term_eq_constraint(self.sym_state_vector[0:,-1], self.params))
                # discretisation time
                self.h = self.h.subs(self.model.dt, self.dt)

        def initialiseInequalityConstraints(self):
                
                self.g = Matrix([])
                
                if(self.ineq_constraint != None):
                        self.g = self.g.col_join(self.ineq_constraint(self.sym_state_vector[0:,0:-1], self.sym_control_vector[0:,0:-1] , self.params))
                if(self.term_ineq_constraint != None):
                        self.g = self.g.col_join(self.term_ineq_constraint(self.sym_state_vector[0:,-1], np.array([[]]), self.params))

        def lineariseConstraints(self):
                # linearisation
                self.A = self.h.jacobian(self.sym_X)
                self.b = -(self.h-self.A*self.sym_X)
                # printing to file
                if(self.code_gen_language == "py"):
                        self.gen_code += "def A_mat(x0, x, u, params):\n\tA = "
                        self.gen_code += sympy.printing.lambdarepr.lambdarepr(self.A)
                        self.gen_code = self.gen_code.replace("ImmutableDenseMatrix", "np.array")
                        self.gen_code += ".astype(float)\n\treturn A\n"
                        self.gen_code += "def b_mat(x0, x, u, params):\n\tb = "
                        self.gen_code += sympy.printing.lambdarepr.lambdarepr(self.b)
                        self.gen_code = self.gen_code.replace("ImmutableDenseMatrix", "np.array")
                        self.gen_code += ".astype(float)\n\treturn b\n"
                        
                elif(self.code_gen_language == "cpp"):
                        if(self.matrix_form == "2d"):
                                self.gen_code += "std::array<std::array<float64_t, "+str(self.A.shape[0])+">,  "+str(self.A.shape[1])+"> A_mat(std::array<float64_t, NX> x0, std::array<std::array<float64_t, NX>, N+1> x, std::array<std::array<float64_t, NU>, N> u, std::array<std::array<float64_t, NP>, N> params)\n{\n"
                                self.gen_code += "\tstd::array<std::array<float64_t,"+str(self.A.shape[0])+">, "+str(self.A.shape[1])+"> A;\n"
                        elif(self.matrix_form == "1drow" or self.matrix_form == "1dcol"):
                                self.gen_code += "float64_t* A_mat(float64_t* x0, float64_t* x, float64_t* u, float64_t* params)\n{\n"
                                self.gen_code += "\tfloat64_t* A = new float64_t["+str(self.A.shape[0]*self.A.shape[1])+"];\n"
                        
                        num_rows = self.A.shape[0]
                        num_cols = self.A.shape[1]
                        
                        for i in range(0, num_rows):
                                for j in range(0, num_cols):
                                        var = str(self.A[i, j])
                                        var = var.replace(", ", "][")
                                        for k in range(0, self.model.NX):
                                                var = var.replace("x0["+str(k)+"][0]", "x0["+str(k)+"]")
                                        if(self.matrix_form == "2d"):
                                                self.gen_code += "\tA["+str(i)+"]["+str(j)+"] = "+var+";\n"
                                        elif(self.matrix_form == "1drow"):
                                                self.gen_code += "\tA["+str(i*num_cols + j)+"] = "+var+";\n"
                                        elif(self.matrix_form == "1dcol"):
                                                self.gen_code += "\tA["+str(i + j*num_rows)+"] = "+var+";\n"                                
                        self.gen_code += "\treturn A;\n"
                        self.gen_code += "}\n"
                        
                        if(self.matrix_form == "2d"):
                                self.gen_code += "std::array<float64_t, "+str(self.b.shape[0])+"> b_mat(std::array<float64_t, NX> x0, std::array<std::array<float64_t, NX>, N+1> x, std::array<std::array<float64_t, NU>, N> u, std::array<std::array<float64_t, NP>, N> params)\n{\n"
                                self.gen_code += "\tstd::array<float64_t, "+str(self.b.shape[0])+"> b;\n"
                        elif(self.matrix_form == "1drow" or self.matrix_form == "1dcol"):
                                self.gen_code += "float64_t* b_mat(float64_t* x0, float64_t* x, float64_t* u, float64_t* params)\n{\n"
                                self.gen_code += "\tfloat64_t* b = new float64_t["+str(self.b.shape[0])+"];\n"
                        
                        for i in range(0, self.b.shape[0]):
                                var = str(self.b[i])
                                var = var.replace(", ", "][")
                                for j in range(0, self.model.NX):
                                        var = var.replace("x0["+str(j)+"][0]", "x0["+str(j)+"]")
                                self.gen_code += "\tb["+str(i)+"] = "+var+";\n"
                        self.gen_code += "\treturn b;\n"
                        self.gen_code += "}\n"
        
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
                
                if(self.code_gen_language == "py"):
                        self.gen_code += "noineq = False\n"
                        self.gen_code+= "def G_mat(x, u, params):\n\tG = np.array("
                        self.gen_code+= np.array2string(np.array(self.G), separator=',')
                        self.gen_code+= ").astype(float)\n\treturn G\n"
                        self.gen_code+= "def h_mat(x, u, params):\n\th = np.array("
                        self.gen_code+= np.array2string(np.array(self.h), separator=',')
                        self.gen_code+= ").astype(float)\n\treturn h\n"
                
                elif(self.code_gen_language == "cpp"):
                        if(self.matrix_form == "2d"):
                                self.gen_code += "std::array<std::array<float64_t, "+str(self.G.shape[0])+">,  "+str(self.G.shape[1])+"> G_mat(std::array<std::array<float64_t, NX>, N+1> x, std::array<std::array<float64_t, NU>, N> u, std::array<std::array<float64_t, NP>, N> params)\n{\n"
                                self.gen_code += "\tstd::array<std::array<float64_t, "+str(self.G.shape[0])+">, "+str(self.G.shape[1])+"> G;\n"
                        elif(self.matrix_form == "1drow" or self.matrix_form == "1dcol"):
                                self.gen_code += "float64_t* G_mat(float64_t* x, float64_t* u, float64_t* params)\n{\n"
                                self.gen_code += "\tfloat64_t* G = new float64_t["+str(self.G.shape[0]*self.G.shape[1])+"];\n"
                        
                        num_rows = self.G.shape[0]
                        num_cols = self.G.shape[1]
                        
                        for i in range(0, num_rows):
                                for j in range(0, num_cols):
                                        var = str(self.G[i, j])
                                        var = var.replace(", ", "][")
                                        if(self.matrix_form == "2d"):
                                                self.gen_code += "\tG["+str(i)+"]["+str(j)+"] = "+var+";\n"
                                        elif(self.matrix_form == "1drow"):
                                                self.gen_code += "\tG["+str(i*num_cols + j)+"] = "+var+";\n"
                                        elif(self.matrix_form == "1dcol"):
                                                self.gen_code += "\tG["+str(i + j*num_rows)+"] = "+var+";\n"
                        self.gen_code += "\treturn G;\n"
                        self.gen_code += "}\n"
                        
                        if(self.matrix_form == "2d"):
                                self.gen_code += "std::array<float64_t, "+str(self.h.shape[0])+"> h_mat(std::array<std::array<float64_t, NX>, N+1> x, std::array<std::array<float64_t, NU>, N> u, std::array<std::array<float64_t, NP>, N> params)\n{\n"
                                self.gen_code += "\tstd::array<float64_t, "+str(self.h.shape[0])+"> h;\n"
                        elif(self.matrix_form == "1drow" or self.matrix_form == "1dcol"):
                                self.gen_code += "float64_t* h_mat(float64_t* x, float64_t* u, float64_t* params)\n{\n"
                                self.gen_code += "\tfloat64_t* h = new float64_t["+str(self.h.shape[0])+"];\n"
                        
                        for i in range(0, self.h.shape[0]):
                                var = str(self.h[i])
                                var = var.replace(", ", "][")
                                self.gen_code += "\th["+str(i)+"] = "+var+";\n"
                        self.gen_code += "\treturn h;\n"
                        self.gen_code += "}\n"

                # lambdifying
                self.G = lambdify([self.sym_X.T], np.squeeze(self.G))
                self.h = lambdify([self.sym_X.T], np.squeeze(self.h))

        def writeGenCode(self):
                if(self.code_gen_language == "py"):
                        Path(self.code_gen_file_name+"/").mkdir(parents=True, exist_ok=True)
                        text_file = open(self.code_gen_file_name+"/"+self.code_gen_file_name.split('/')[1]+".py", "w")
                        text_file.write(self.gen_code)
                        text_file.close()
                elif(self.code_gen_language == "cpp"):
                        Path(self.code_gen_file_name+"/").mkdir(parents=True, exist_ok=True)
                        text_file = open(self.code_gen_file_name+"/"+self.code_gen_file_name.split('/')[1]+".hpp", "w")                

                        for i in range (0, self.N+1):
                                for j in range(0, self.model.NX):
                                        if(self.matrix_form == "1drow"):
                                                self.gen_code = self.gen_code.replace("x["+str(j)+"]["+str(i)+"]", "x["+str((self.model.NX)*j + i)+"]")
                                                self.gen_code = self.gen_code.replace("x_ref["+str(j)+"]["+str(i)+"]", "x_ref["+str((self.model.NX)*j + i)+"]")
                                        elif(self.matrix_form == "1dcol"):
                                                self.gen_code = self.gen_code.replace("x["+str(j)+"]["+str(i)+"]", "x["+str((self.N+1)*i + j)+"]")
                                                self.gen_code = self.gen_code.replace("x_ref["+str(j)+"]["+str(i)+"]", "x_ref["+str((self.N+1)*i + j)+"]")
                                if (i == self.N):
                                        continue
                                        
                                for j in range(0, self.model.NU):
                                        if(self.matrix_form == "1drow"):
                                                self.gen_code = self.gen_code.replace("u["+str(j)+"]["+str(i)+"]", "u["+str((self.model.NU)*j + i)+"]")
                                                self.gen_code = self.gen_code.replace("u_ref["+str(j)+"]["+str(i)+"]", "u_ref["+str((self.model.NU)*j + i)+"]")
                                        elif(self.matrix_form == "1dcol"):
                                                self.gen_code = self.gen_code.replace("u["+str(j)+"]["+str(i)+"]", "u["+str((self.N)*i + j)+"]")
                                                self.gen_code = self.gen_code.replace("u_ref["+str(j)+"]["+str(i)+"]", "u_ref["+str((self.N)*i + j)+"]")
                                for j in range(0, self.model.NP):
                                        if(self.matrix_form == "1drow"):
                                                self.gen_code.replace("params["+str(j)+"]["+str(i)+"]", "params["+str((self.model.NP)*j + i)+"]")
                                        elif(self.matrix_form == "1dcol"):
                                                self.gen_code = self.gen_code.replace("params["+str(j)+"]["+str(i)+"]", "params["+str((self.N)*i + j)+"]")
                        
                        text_file.write(self.gen_code)
                        text_file.close()                        
