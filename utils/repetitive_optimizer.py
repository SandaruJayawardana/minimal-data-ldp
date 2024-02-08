'''
    This class contain the converging algorithm or equivalent optimization problem. 
'''

import numpy as np
from utils.util_functions import *
from utils.randomized_response import random_response_mechanism

class Repetitive_optimizer:
    def __init__(self, prior_dist, normalized_err_matrix, TOLERANCE_MARGIN = 0.01, APPROXIMATION = "LINEAR", STATE_COUNT = 4):
        self.TOLERANCE_MARGIN = TOLERANCE_MARGIN
        self.APPROXIMATION = APPROXIMATION
        self.STATE_COUNT = STATE_COUNT

        if len(prior_dist) != self.STATE_COUNT:
            assert("Prior distribution and state count do not match")
        self.prior_dist = prior_dist

        if validate_err_matrix(normalized_err_matrix, self.STATE_COUNT):
            self.normalized_err_matrix = normalized_err_matrix

        self.normalized_err_matrix = normalized_err_matrix
        self.__optimal_eps = -1
        self.__optimal_mechanism = -1
        self.history = []
        self.max_var = 5
        self.flag =False

    def __get_random_response_eps__(self, min_utility_err):
        rr_eps = np.log(((np.sum(np.sum(self.normalized_err_matrix, axis=0)*self.prior_dist))/min_utility_err) - (self.STATE_COUNT - 1))
        return rr_eps
    
    def __converge__(self, next_eps, last_eps, last_util_err, min_utility_err, optimizer):
        if next_eps < 0:
            next_eps = 0
            if not(self.flag):
                self.flag = True
                print("Negative eps, first")
            else:
                self.__optimal_mechanism = optimizer.get_mechanism(next_eps, min_utility_err = min_utility_err,  max_var = self.max_var)
                # print(" Optimized ", self.__optimal_mechanism)
                print("Negative, second, optimal_eps ", next_eps)
                return self.__optimal_mechanism
        
        if next_eps > 10:
            next_eps = last_eps
            self.__optimal_mechanism = optimizer.get_mechanism(next_eps, min_utility_err = min_utility_err,  max_var = self.max_var)
            # print(" Optimized ", self.__optimal_mechanism)
            print("Flag, optimal_eps ", next_eps)
            return self.__optimal_mechanism
        
        next_util_err = optimizer.get_utility(next_eps, min_utility_err = min_utility_err,  max_var = self.max_var)
        self.history.append({"eps": next_eps, "util": next_util_err})
        # print("next_eps ", next_eps, " next_util_err ", next_util_err, " last_eps ", last_eps, " last_util_err ", last_util_err)

        if abs(min_utility_err - next_util_err) < self.TOLERANCE_MARGIN:
            self.__optimal_mechanism = optimizer.get_mechanism(next_eps, min_utility_err = min_utility_err,  max_var = self.max_var)
            # print("Optimal eps ", next_eps, " Utility err ", next_util_err)
            # print(" Optimized ", self.__optimal_mechanism)
            print("optimal_eps ", next_eps)
            return self.__optimal_mechanism
        
        if self.APPROXIMATION == "LINEAR":
            start_eps = (min_utility_err - last_util_err)*(next_eps - last_eps)/(next_util_err - last_util_err) + last_eps
            # print("(min_utility_err - last_util_err) ", (min_utility_err - last_util_err))
            # print("(next_util_err - last_util_err) ", (next_util_err - last_util_err))
            # print("(next_eps - last_eps) ", (next_eps - last_eps))
            # print(" Interpolation ", start_eps)
        else:
            assert False, f"Unknown Approximation {self.APPROXIMATION}"

        return self.__converge__(next_eps=start_eps, last_eps=next_eps, last_util_err=next_util_err, min_utility_err=min_utility_err, optimizer=optimizer)
    
    def __get_rr_mechanism__(self, eps):
        return random_response_mechanism(alphabet_size=self.STATE_COUNT, eps=eps)
    
    def get_eps(self):
        if self.__optimal_eps < 0:
            assert("Not optimized yet. Run 'get_optimal_mechanism()'")
        return self.__optimal_eps

    def get_optimal_mechanism(self, min_utility_err, optimizer):
        rr_eps = self.__get_random_response_eps__(min_utility_err)

        # print("rr_eps ", rr_eps)

        optimal_util = optimizer.get_utility(eps=rr_eps, min_utility_err = min_utility_err,  max_var = self.max_var)
        self.history.append({"eps": rr_eps, "util": optimal_util})
        # print("optimal_util ", optimal_util)
        # print("abs(min_utility_err - optimal_util)", abs(min_utility_err - optimal_util), "Max allowed error", self.TOLERANCE_MARGIN, abs(min_utility_err - optimal_util) < self.TOLERANCE_MARGIN)
        if abs(min_utility_err - optimal_util) < self.TOLERANCE_MARGIN:
            self.__optimal_mechanism = self.__get_rr_mechanism__(rr_eps)
            return self.__optimal_mechanism
        
        return self.__converge__(next_eps = rr_eps/2, last_eps = rr_eps, last_util_err = optimal_util, min_utility_err = min_utility_err, optimizer = optimizer)
