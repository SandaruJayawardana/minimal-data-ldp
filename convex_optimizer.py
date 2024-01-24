'''
    Convex optimizer using CVXPY.
'''

import cvxpy as cp
import numpy as np
from util_functions import *

class Optimizer:
    def __init__(self, prior_dist, normalized_objective_err_matrix, TOLERANCE_MARGIN = 0.01, APPROXIMATION = "LINEAR", STATE_COUNT = 4, solver = "SCS", is_kl_div = True, ALPHA=0.01):
        self.solver = solver
        self.TOLERANCE_MARGIN = TOLERANCE_MARGIN
        self.APPROXIMATION = APPROXIMATION
        self.STATE_COUNT = STATE_COUNT
        self.ALPHA = ALPHA
        self.__is_kl_div = is_kl_div

        if validate_distribution(prior_dist, self.STATE_COUNT):
            self.prior_dist = prior_dist

        if validate_err_matrix(normalized_objective_err_matrix, self.STATE_COUNT):
            self.normalized_objective_err_matrix = normalized_objective_err_matrix

        self.__optimal_mechanism = -1
        self.__constraints = []
    
    def add_constraint_err_matrix(self, normilized_constraint_err_matrix):
        self.__constraints.append(normilized_constraint_err_matrix)

    def get_utility(self, eps, min_utility_err = 0,  max_var = 0):
        solution = self.optimize(eps=eps, min_utility_err=min_utility_err,  max_var=max_var)
        return solution["utility"]

    def get_mechanism(self, eps, min_utility_err = 0,  max_var = 0):
        print(eps)
        solution = self.optimize(eps=eps, min_utility_err=min_utility_err, max_var=max_var)
        self.__optimal_mechanism = solution["mechanism"]
        return self.__optimal_mechanism
    
    # #original
    # def optimize(self, eps, min_utility_err, max_var):
    #     # max_var = 0.4
    #     # Define the optimization variable
    #     X = cp.Variable((self.STATE_COUNT, self.STATE_COUNT))

    #     # Objective function
    #     objective = cp.Minimize(np.transpose(self.prior_dist)@cp.sum(cp.multiply(X, self.normalized_objective_err_matrix), axis=1)) #cp.Minimize((X@np.transpose(U))@Z)
    #     # objective = cp.Minimize((np.transpose(self.prior_dist)@cp.square(cp.sum(cp.multiply(X, self.normalized_objective_err_matrix), axis=1))-min_utility_err**2))
    #     # Constraints
    #     X_T = X.T
    #     Y = X_T@self.prior_dist
    #     constraints = [cp.sum(X, axis=1) == 1,  # Each row sums to 1
    #                 X >= 0.000001] # Non-negative elements
        
    #     # if max_var != 0:
    #     #     # constraints.append(np.transpose(self.prior_dist)@cp.sum(cp.multiply(X, self.normalized_objective_err_matrix), axis=1)<= max_var)
    #     #     constraints.append((np.transpose(self.prior_dist)@cp.square(cp.sum(cp.multiply(X, self.normalized_objective_err_matrix), axis=1))-min_utility_err**2) <= max_var)
    #         # print("Add constraint ", constraints[-1])
    #     if self.__is_kl_div:  # KL divergence
    #         constraints.append(cp.sum(cp.kl_div(Y, self.prior_dist)) <= self.ALPHA)

    #     # Adding KL divergence constraints for each row
    #     for i in range(self.STATE_COUNT):
    #         for j in range(self.STATE_COUNT):
    #             for k in range(self.STATE_COUNT):
    #                 if j == k:
    #                     continue
    #                 constraints.append(X[j, i] - np.exp(eps)*X[k, i] <= 0)
    #                 # print(f"X[{j}, {i}] - np.exp(EPS)*X[{k}, {i}] <= 0")

    #     # Define and solve the problem
    #     problem = cp.Problem(objective, constraints)

    #     if self.solver == "SCS":
    #         problem.solve(solver=cp.SCS, verbose=False)
    #     elif self.solver == "MOSEK":
    #         problem.solve(solver=cp.MOSEK, verbose=False)
    #     else:
    #         assert(f"Unkwon solver {self.solver}")

    #     # Output the optimized matrix
    #     matrix = np.maximum(np.array(X.value), 0)
    #     row_sums = matrix.sum(axis=1, keepdims=True)

    #     # print("var ", np.dot(np.transpose(self.prior_dist), np.square(np.sum(np.multiply(matrix/row_sums, self.normalized_objective_err_matrix), axis=1))) - 0.4**2)
    #     # print("Expected val ", np.dot(np.transpose(self.prior_dist), np.sum(np.multiply(matrix/row_sums, self.normalized_objective_err_matrix), axis=1)))
    #     # print("Optimized Matrix P:\n", X.value)
    #     # print("Original Distribution Z:\n", Z)
    #     # print("Perturbed Distribution Z:\n", np.matmul(np.transpose(Z),(matrix/row_sums)))
    #     return {"utility": objective.value, "mechanism": matrix/row_sums}


#   TV original
    def optimize(self, eps, min_utility_err, max_var):
        # max_var = 0.4
        # Define the optimization variable
        X = cp.Variable((self.STATE_COUNT, self.STATE_COUNT))

        # Objective function
        objective = cp.Minimize(np.transpose(self.prior_dist)@cp.sum(cp.multiply(X, self.normalized_objective_err_matrix), axis=1)) #cp.Minimize((X@np.transpose(U))@Z)
        # objective = cp.Minimize((np.transpose(self.prior_dist)@cp.square(cp.sum(cp.multiply(X, self.normalized_objective_err_matrix), axis=1))-min_utility_err**2))
        # Constraints
        X_T = X.T
        Y = X_T@self.prior_dist
        constraints = [cp.sum(X, axis=1) == 1,  # Each row sums to 1
                    X >= 0.000001] # Non-negative elements
        
        # if max_var != 0:
        #     # constraints.append(np.transpose(self.prior_dist)@cp.sum(cp.multiply(X, self.normalized_objective_err_matrix), axis=1)<= max_var)
        #     constraints.append((np.transpose(self.prior_dist)@cp.square(cp.sum(cp.multiply(X, self.normalized_objective_err_matrix), axis=1))-min_utility_err**2) <= max_var)
            # print("Add constraint ", constraints[-1])
        if self.__is_kl_div:  # KL divergence
            # constraints.append(cp.sum(cp.kl_div(Y, self.prior_dist)) <= self.ALPHA)
            constraints.append((0.5 * cp.sum(cp.abs(Y - self.prior_dist))) <= self.ALPHA)

        # Adding KL divergence constraints for each row
        for i in range(self.STATE_COUNT):
            for j in range(self.STATE_COUNT):
                for k in range(self.STATE_COUNT):
                    if j == k:
                        continue
                    constraints.append(X[j, i] - np.exp(eps)*X[k, i] <= 0)
                    # print(f"X[{j}, {i}] - np.exp(EPS)*X[{k}, {i}] <= 0")

        # Define and solve the problem
        problem = cp.Problem(objective, constraints)

        if self.solver == "SCS":
            problem.solve(solver=cp.SCS, verbose=False)
        elif self.solver == "MOSEK":
            problem.solve(solver=cp.MOSEK, verbose=False)
        else:
            assert(f"Unkwon solver {self.solver}")

        # Output the optimized matrix
        matrix = np.maximum(np.array(X.value), 0)
        row_sums = matrix.sum(axis=1, keepdims=True)

        # print("var ", np.dot(np.transpose(self.prior_dist), np.square(np.sum(np.multiply(matrix/row_sums, self.normalized_objective_err_matrix), axis=1))) - 0.4**2)
        # print("Expected val ", np.dot(np.transpose(self.prior_dist), np.sum(np.multiply(matrix/row_sums, self.normalized_objective_err_matrix), axis=1)))
        # print("Optimized Matrix P:\n", X.value)
        # print("Original Distribution Z:\n", Z)
        # print("Perturbed Distribution Z:\n", np.matmul(np.transpose(Z),(matrix/row_sums)))
        return {"utility": objective.value, "mechanism": matrix/row_sums}


    # def optimize(self, eps, min_utility_err, max_var):
    #     # max_var = 0.4
    #     # Define the optimization variable
    #     X = cp.Variable((self.STATE_COUNT, self.STATE_COUNT))

    #     # Objective function
    #     objective = cp.Minimize(np.transpose(self.prior_dist)@cp.sum(cp.multiply(X, self.normalized_objective_err_matrix), axis=1)) #cp.Minimize((X@np.transpose(U))@Z)
    #     # objective = cp.Minimize((np.transpose(self.prior_dist)@cp.square(cp.sum(cp.multiply(X, self.normalized_objective_err_matrix), axis=1))-min_utility_err**2))
    #     # Constraints
    #     X_T = X.T
    #     Y = X_T@self.prior_dist
    #     constraints = [cp.sum(X, axis=1) == 1,  # Each row sums to 1
    #                 X >= 0.000001] # Non-negative elements
        
    #     # if max_var != 0:
    #     #     # constraints.append(np.transpose(self.prior_dist)@cp.sum(cp.multiply(X, self.normalized_objective_err_matrix), axis=1)<= max_var)
    #     #     constraints.append((np.transpose(self.prior_dist)@cp.square(cp.sum(cp.multiply(X, self.normalized_objective_err_matrix), axis=1))-min_utility_err**2) <= max_var)
    #         # print("Add constraint ", constraints[-1])
    #     if self.__is_kl_div:  # KL divergence
    #         # constraints.append(cp.sum(cp.kl_div(Y, self.prior_dist)) <= self.ALPHA)
    #         constraints.append((0.5 * cp.sum(cp.abs(Y - self.prior_dist))) <= self.ALPHA)

    #     # Adding KL divergence constraints for each row
    #     # for i in range(self.STATE_COUNT):
    #     #     for j in range(self.STATE_COUNT):
    #     #         for k in range(self.STATE_COUNT):
    #     #             if j == k:
    #     #                 continue
    #     #             constraints.append(X[j, i] - np.exp(eps)*X[k, i] <= 0)
    #                 # print(f"X[{j}, {i}] - np.exp(EPS)*X[{k}, {i}] <= 0")
            
    #     constraints.append((X[0, 0]+X[0, 1]+X[1, 0]+X[1, 1]) - np.exp(eps)*(X[2, 0]+X[2, 1]+X[3, 0]+X[3, 1]) <= 0) 
    #     constraints.append((X[2, 0]+X[2, 1]+X[3, 0]+X[3, 1]) - np.exp(eps)*(X[0, 0]+X[0, 1]+X[1, 0]+X[1, 1]) <= 0) 
    #     constraints.append((X[0, 2]+X[0, 3]+X[1, 2]+X[1, 3]) - np.exp(eps)*(X[2, 2]+X[2, 3]+X[3, 2]+X[3, 3]) <= 0) 
    #     constraints.append((X[2, 2]+X[2, 3]+X[3, 2]+X[3, 3]) - np.exp(eps)*(X[0, 2]+X[0, 3]+X[1, 2]+X[1, 3]) <= 0) 

    #     constraints.append((X[0, 0]+X[0, 2]+X[2, 0]+X[2, 2]) - np.exp(eps)*(X[1, 0]+X[1, 2]+X[3, 0]+X[3, 2]) <= 0) 
    #     constraints.append((X[1, 0]+X[1, 2]+X[3, 0]+X[3, 2]) - np.exp(eps)*(X[0, 0]+X[0, 2]+X[2, 0]+X[2, 2]) <= 0) 
    #     constraints.append((X[0, 1]+X[0, 3]+X[2, 1]+X[2, 3]) - np.exp(eps)*(X[1, 1]+X[1, 3]+X[3, 1]+X[3, 3]) <= 0) 
    #     constraints.append((X[1, 1]+X[1, 3]+X[3, 1]+X[3, 3]) - np.exp(eps)*(X[0, 1]+X[0, 3]+X[2, 1]+X[2, 3]) <= 0) 

    #     # Define and solve the problem
    #     problem = cp.Problem(objective, constraints)

    #     if self.solver == "SCS":
    #         problem.solve(solver=cp.SCS, verbose=False)
    #     elif self.solver == "MOSEK":
    #         problem.solve(solver=cp.MOSEK, verbose=False)
    #     else:
    #         assert(f"Unkwon solver {self.solver}")

    #     # Output the optimized matrix
    #     matrix = np.maximum(np.array(X.value), 0)
    #     row_sums = matrix.sum(axis=1, keepdims=True)

    #     # print("var ", np.dot(np.transpose(self.prior_dist), np.square(np.sum(np.multiply(matrix/row_sums, self.normalized_objective_err_matrix), axis=1))) - 0.4**2)
    #     # print("Expected val ", np.dot(np.transpose(self.prior_dist), np.sum(np.multiply(matrix/row_sums, self.normalized_objective_err_matrix), axis=1)))
    #     # print("Optimized Matrix P:\n", X.value)
    #     # print("Original Distribution Z:\n", Z)
    #     # print("Perturbed Distribution Z:\n", np.matmul(np.transpose(Z),(matrix/row_sums)))
    #     return {"utility": objective.value, "mechanism": matrix/row_sums}



    # def optimize2(self, eps, min_utility_err, max_var):

    #     expected_error_matrix = np.ones((self.STATE_COUNT, self.STATE_COUNT))
    #     sum_row_values = np.ones((self.STATE_COUNT))

    #     for i in range(len(self.normalized_objective_err_matrix[0])):
    #         expected_error_matrix[:,i] = self.prior_dist*self.normalized_objective_err_matrix[:,i]
    #         sum_row_values[i] = np.sum(expected_error_matrix[:,i])
    #         expected_error_matrix[:,i] /= np.max(expected_error_matrix[:,i])
    #         for j in range(self.STATE_COUNT):
    #             expected_error_matrix[j,i] = 1 if expected_error_matrix[j,i] == 0 else 1/np.exp(eps) #(1-expected_error_matrix[:,i]) + expected_error_matrix[:,i]/np.exp(eps)
    #     print("expected_error_matrix", expected_error_matrix)

    #     row_values_list = []
    #     for i in range(self.STATE_COUNT):
    #         row_values_list.append(cp.Constant(expected_error_matrix[i,:]))
        
    #     print(sum_row_values)
    #     sum_row_values = cp.Constant(sum_row_values*np.sum(expected_error_matrix, axis=0))
    #     print(sum_row_values)
    #     # max_var = 0.4
    #     # Define the optimization variable
    #     X_new = cp.Variable((self.STATE_COUNT))
    #     X = cp.Variable((self.STATE_COUNT, self.STATE_COUNT))

    #     # AAA = cp.Variable((self.STATE_COUNT))

    #     # BBB = [expected_error_matrix[i, :] * AAA for i in range(expected_error_matrix.shape[0])]
    #     # X = BBB
    #     # Objective function

    #     objective = cp.Minimize(cp.sum(cp.multiply(sum_row_values,X_new)))
    #     print("sum_row_values", sum_row_values)

    #     # objective = cp.Minimize(np.transpose(self.prior_dist)@cp.sum(cp.multiply(X, self.normalized_objective_err_matrix), axis=1)) #cp.Minimize((X@np.transpose(U))@Z)
        
        
    #     # objective = cp.Minimize((np.transpose(self.prior_dist)@cp.square(cp.sum(cp.multiply(X, self.normalized_objective_err_matrix), axis=1))-min_utility_err**2))
    #     # Constraints
    #     # X_T = X.T
    #     # Y = X_T@self.prior_dist
    #     # constraints = [cp.sum(X, axis=1) == 1,  # Each row sums to 1
    #     #             X >= 0.000001] # Non-negative elements

    #     constraints = [X_new >= 0.000001] # Non-negative elements

    #     for i in range(self.STATE_COUNT):
    #         constraints.append(cp.sum(cp.multiply(X_new,row_values_list[i])) - 1.1 <= 0) # Each row sums to 1
    #         constraints.append(cp.sum(cp.multiply(X_new,row_values_list[i])) - 0.9 >= 0) # Each row sums to 1
        
    #     # if max_var != 0:
    #     #     # constraints.append(np.transpose(self.prior_dist)@cp.sum(cp.multiply(X, self.normalized_objective_err_matrix), axis=1)<= max_var)
    #     #     constraints.append((np.transpose(self.prior_dist)@cp.square(cp.sum(cp.multiply(X, self.normalized_objective_err_matrix), axis=1))-min_utility_err**2) <= max_var)
    #         # print("Add constraint ", constraints[-1])
    #     # if self.__is_kl_div:  # KL divergence
    #     #     constraints.append(cp.sum(cp.kl_div(Y, self.prior_dist)) <= self.ALPHA)

    #     # Adding KL divergence constraints for each row
    #     # for i in range(self.STATE_COUNT):
    #     #     for j in range(self.STATE_COUNT):
    #     #         for k in range(self.STATE_COUNT):
    #     #             if j == k:
    #     #                 continue
    #     #             constraints.append(X[j, i] - np.exp(eps)*X[k, i] <= 0)
    #                 # print(f"X[{j}, {i}] - np.exp(EPS)*X[{k}, {i}] <= 0")

    #     # Define and solve the problem
    #     problem = cp.Problem(objective, constraints)

    #     if self.solver == "SCS":
    #         problem.solve(solver=cp.SCS, verbose=False)
    #     elif self.solver == "MOSEK":
    #         problem.solve(solver=cp.MOSEK, verbose=False)
    #     else:
    #         assert(f"Unkwon solver {self.solver}")

    #     # Output the optimized matrix
    #     print("X_new.value", X_new.value)
    #     matrix = np.zeros((self.STATE_COUNT, self.STATE_COUNT))
    #     for i in range(self.STATE_COUNT):
    #         matrix[i,:] = np.array(X_new.value)*expected_error_matrix[i,:]
    #     matrix = np.maximum(matrix, 0)
        
    #     row_sums = matrix.sum(axis=1, keepdims=True)
    #     print("matrix", matrix)
        
    #     # print("var ", np.dot(np.transpose(self.prior_dist), np.square(np.sum(np.multiply(matrix/row_sums, self.normalized_objective_err_matrix), axis=1))) - 0.4**2)
    #     # print("Expected val ", np.dot(np.transpose(self.prior_dist), np.sum(np.multiply(matrix/row_sums, self.normalized_objective_err_matrix), axis=1)))
    #     # print("Optimized Matrix P:\n", X.value)
    #     # print("Original Distribution Z:\n", Z)
    #     # print("Perturbed Distribution Z:\n", np.matmul(np.transpose(Z),(matrix/row_sums)))
    #     return {"utility": objective.value, "mechanism": matrix/row_sums}
