import numpy as np
from privacy_mechanism import *
from util_functions import *
from convex_optimizer import *

class Optimized_Randomized_Response(Privacy_Mechanism):
    
    def __init__(self, prior_dist, STATE_COUNT, INPUT_ALPHABET = [], normalized_objective_err_matrix = [], 
                 TOLERANCE_MARGIN = 0.01, APPROXIMATION = "LINEAR", solver = "SCS", is_kl_div = True, ALPHA=0.01):
        self.STATE_COUNT = STATE_COUNT

        if len(INPUT_ALPHABET) != STATE_COUNT:
            self.INPUT_ALPHABET = str(range(STATE_COUNT))
        elif validate_alphabet(INPUT_ALPHABET):
            self.INPUT_ALPHABET = INPUT_ALPHABET
        
        if validate_distribution(prior_dist, self.STATE_COUNT):
            self.prior_dist = prior_dist
        
        self.__mechanism = 0
        self.__eps = -1
        self.normalized_objective_err_matrix = normalized_objective_err_matrix
        self.optimizer = Optimizer(self.prior_dist, normalized_objective_err_matrix, TOLERANCE_MARGIN, 
                                   APPROXIMATION, STATE_COUNT, solver, is_kl_div, ALPHA)
    
    def get_mechanism(self, eps):
        if self.__eps == eps:
            return self.__mechanism
        self.__eps = eps
        self.__mechanism = self.optimizer.get_mechanism(eps=eps)
        return self.__mechanism

    def get_name(self):
        return "Optimized Random Response"
