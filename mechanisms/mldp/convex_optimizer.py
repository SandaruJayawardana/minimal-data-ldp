'''
    Convex optimizer using CVXPY.
'''

import cvxpy as cp
import numpy as np
from utils.util_functions import *

class Optimizer:
    def __init__(self, prior_dist, normalized_objective_err_matrix, TOLERANCE_MARGIN = 0.01, APPROXIMATION = "LINEAR", STATE_COUNT = 4, solver = "SCS", is_kl_div = True, ALPHA=0.01, alphabet = []):
        self.solver = solver
        self.TOLERANCE_MARGIN = TOLERANCE_MARGIN
        self.APPROXIMATION = APPROXIMATION
        self.STATE_COUNT = STATE_COUNT
        self.ALPHA = ALPHA
        self.__is_kl_div = is_kl_div
        self.alphabet = alphabet

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
        # print(eps)
        solution = self.optimize(eps=eps, min_utility_err=min_utility_err, max_var=max_var)
        self.__optimal_mechanism = solution["mechanism"]
        return self.__optimal_mechanism
    
    #original
    def optimize(self, eps, min_utility_err, max_var):
        X = cp.Variable((self.STATE_COUNT, self.STATE_COUNT))

        # Objective function
        objective = cp.Minimize(np.transpose(self.prior_dist)@cp.sum(cp.multiply(X, self.normalized_objective_err_matrix), axis=1)) #cp.Minimize((X@np.transpose(U))@Z)
       
        # Constraints
        X_T = X.T
        Y = X_T@self.prior_dist
        constraints = [cp.sum(X, axis=1) == 1,  # Each row sums to 1
                    X >= 0.000001] # Non-negative elements
  
        if self.__is_kl_div:  # KL divergence
            constraints.append(cp.sum(cp.kl_div(Y, self.prior_dist)) <= self.ALPHA)

        # Adding KL divergence constraints for each row
        for i in range(self.STATE_COUNT):
            for j in range(self.STATE_COUNT):
                for k in range(self.STATE_COUNT):
                    if j == k:
                        continue
                    constraints.append(X[j, i] - np.exp(eps)*X[k, i] <= 0)

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

        return {"utility": objective.value, "mechanism": matrix/row_sums}


#   TV original
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
    #         scs_parameters = {
    #                             # 'max_iters': 6000,        # Maximum number of iterations
    #                             'eps': 1e-4,              # Convergence tolerance
    #                             # 'alpha': 1.5,             # Relaxation parameter
    #                             # 'rho_x': 1e-5,            # Balances primal and dual progress
    #                             # 'scale': 1,               # Scaling parameter for data matrix
    #                             'warm_start': False,       # Use solution from previous call as a starting point
    #                             'acceleration_lookback': 1, # Memory of the acceleration method
    #                             # 'normalize': True,        # Automatically scale the data
    #                             'verbose': False           # Print out progress information
    #                         }
            
    #         problem.solve(solver=cp.SCS, **scs_parameters)
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
