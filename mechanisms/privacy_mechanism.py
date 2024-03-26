'''
    This is a abstract class for general privacy mechanisms.
'''

import numpy as np

class Privacy_Mechanism:

    def __str__(self):
        if self.__mechanism  == 0:
            return "Mechanism is not generated yet. Run get_mechanism()."
        return f"Mechanism for eps = {self.__eps} = {self.__mechanism}"
    
    def get_mechanism(self):
        '''
            Output format:
            [[p(y0|x0), p(y1|x0), p(y2|x0)], [p(y0|x1), p(y1|x1), p(y2|x1)], [p(y0|x2), p(y1|x2), p(y2|x2)]]
        '''
        assert False, "cal_probabilities is not implemented"

    def get_eps(self):
        return self.__eps

    def gen_random_output(self, actual_value, eps, is_index = False):
        # print(f"{actual_value}  alphabet {self.INPUT_ALPHABET}")
        try:
            index_of_actual_value = self.INPUT_ALPHABET.index(actual_value)
        except ValueError:
            print(f"{actual_value} is not in the alphabet {self.INPUT_ALPHABET}")
            
        prob_vec = self.get_mechanism(eps)[index_of_actual_value,:]
        random_ = np.random.choice(len(self.INPUT_ALPHABET) if is_index else self.INPUT_ALPHABET, 1, p=prob_vec)
        return random_
    
    def get_expected_utility_error(self, eps, input_probability = None):
        # if input_probability == None:
        #     assert self.prior_dist != None, "Default distribution not available. Pass a distribution."
        #     input_probability = self.prior_dist
        # assert self.normalized_objective_err_matrix != None, "Nomalized Error matrix not available"

        return np.sum(input_probability * np.sum(self.get_mechanism(eps) * self.normalized_objective_err_matrix, axis= 1))

    def get_name(self):
        assert False, "get_name is not implemented"
    