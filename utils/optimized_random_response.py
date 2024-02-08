import numpy as np
from utils.privacy_mechanism import *
from utils.util_functions import *
from utils.convex_optimizer import *
from utils.randomized_response import Randomized_Response

class Optimized_Randomized_Response(Privacy_Mechanism):
    
    def __init__(self, prior_dist, STATE_COUNT, INPUT_ALPHABET = [], normalized_objective_err_matrix = [], 
                 TOLERANCE_MARGIN = 0.01, APPROXIMATION = "LINEAR", solver = "SCS", is_kl_div = True, ALPHA=0.01, accelerate_from_rr = False):
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
                                   APPROXIMATION, STATE_COUNT, solver, is_kl_div, ALPHA, alphabet=INPUT_ALPHABET)

        self.accelerate_from_rr = accelerate_from_rr # If optimal solution close to rr, future eps values will be adopted from rr
        
        if accelerate_from_rr:
            self.close_to_rr = False
            self.eps_at_close_to_rr = 100
            self.random_response_mechanism = Randomized_Response(STATE_COUNT=STATE_COUNT, INPUT_ALPHABET=INPUT_ALPHABET, normalized_objective_err_matrix=normalized_objective_err_matrix)
    
    def get_mechanism(self, eps):
        if self.__eps == eps:
            return self.__mechanism
        if self.accelerate_from_rr:
            if self.close_to_rr and eps > self.eps_at_close_to_rr:
                self.__mechanism = self.random_response_mechanism.get_mechanism(eps=eps)
                self.__eps = eps
                return self.__mechanism
            
        self.__eps = eps
        self.__mechanism = self.optimizer.get_mechanism(eps=eps)

        if self.accelerate_from_rr and not(self.close_to_rr):
            rr_mechanism = self.random_response_mechanism.get_mechanism(eps=eps)
            l2_error = np.linalg.norm((self.__mechanism  - rr_mechanism))
            print("l2_error ", l2_error)
            if l2_error < 0.001:
                self.close_to_rr = True
                self.eps_at_close_to_rr = eps

        return self.__mechanism

    def get_name(self):
        return "Optimized Random Response"
