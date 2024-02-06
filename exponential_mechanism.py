import numpy as np
from privacy_mechanism import *
from util_functions import *
import matplotlib.pyplot as plt
# import seaborn as sns

def exp_value(eps, score, sensitivity):
    return np.exp(eps * score / (2 * sensitivity))

class Exponential_mechanism(Privacy_Mechanism):
    
    def __init__(self, prior_dist, STATE_COUNT, INPUT_ALPHABET = [], normalized_objective_err_matrix = None, only_err_matrix = True):
        self.STATE_COUNT = STATE_COUNT

        if len(INPUT_ALPHABET) != STATE_COUNT:
            self.INPUT_ALPHABET = str(range(STATE_COUNT))
        elif validate_alphabet(INPUT_ALPHABET):
            self.INPUT_ALPHABET = INPUT_ALPHABET
        
        validate_err_matrix(err_matrix=normalized_objective_err_matrix, dim=STATE_COUNT)
        
        self.__mechanism = 0
        self.__eps = -1
        self.normalized_objective_err_matrix = normalized_objective_err_matrix
        self.prior_dist = prior_dist
        self.__only_error_mtrix = only_err_matrix
        self.sensitivity = 1

    
    def __exponential_mechanism__(self):
        mechanism = np.zeros((self.STATE_COUNT, self.STATE_COUNT))
        
        if self.__only_error_mtrix:
            q_function = np.ones((self.STATE_COUNT, self.STATE_COUNT)) - self.normalized_objective_err_matrix
        else:
            q_function = np.ones((self.STATE_COUNT, self.STATE_COUNT)) - self.normalized_objective_err_matrix*np.transpose(self.prior_dist)
        q_function = q_function*5 #/np.max(q_function)
        self.sensitivity = np.max(q_function) - np.min(q_function)

        for i in range(self.STATE_COUNT):
            probabilities = [exp_value(eps=self.__eps, score=score, sensitivity=self.sensitivity) for score in q_function[i,:]]
            mechanism[i,:] = probabilities / np.linalg.norm(probabilities, ord=1)
        # sns.heatmap(mechanism)
        # plt.show()
        return mechanism
        

    def get_mechanism(self, eps):
        if self.__eps == eps:
            return self.__mechanism
        self.__eps = eps
        self.__mechanism = self.__exponential_mechanism__()
        return self.__mechanism

    def get_name(self):
        return "Exponential Mechanism"
