import numpy as np
from mechanisms.privacy_mechanism import *
from utils.normalize_error_matrix import Normalize_error_matrix
from utils.util_functions import *
from mechanisms.mldp.convex_optimizer import *
from mechanisms.mldp.mldp import *
from mechanisms.k_rr.k_rr_mechanism import Randomized_Response



            
# create_optimal_mechnism_dict(alphabet=ALL_ALPHABET, alphabet_dict=alphabet_dict, prior_dist=random_dist, err_type="0_1")

class Approximated_Optimal_Randomized_Response(Privacy_Mechanism):
    
    def __init__(self, prior_dist, STATE_COUNT, NUM_ATTRIBUTES, INPUT_ALPHABET = [], normalized_objective_err_matrix = [], 
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

        self.__mechanism_list = []
        
        for i in range(NUM_ATTRIBUTES):
            mechanism_dict = {}
            individual_alphabet = alphabet_dict[str(i)]
            state_count = len(individual_alphabet)
            normalize_error_matrix = Normalize_error_matrix(attribute_list=[str(i)], alphabet=individual_alphabet, priority_dict=priority_dict, alphabet_dict=alphabet_dict, err_type=err_type)
            err_matrix = normalize_error_matrix.normalized_error_matrix
       
            for index_j, j in enumerate(alphabet):
                predessor_index_list = j[:i]
                if len(predessor_index_list) == 0:
                    if uniform:
                        conditional_prior_dist = np.ones(len(individual_alphabet))/len(individual_alphabet)
                    else:
                        conditional_prior_dist = get_conditional_dist(i, predessor_index_list, (individual_alphabet), alphabet, prior_dist)
                    # err_matrix_1 = err_matrix*(1- conditional_prior_dist) #(1/(conditional_prior_dist+0.00000001))
                    # err_matrix_1 = err_matrix_1 / np.max(err_matrix_1)
                    # sns.heatmap(err_matrix_1)
                    # plt.show()
                    print("conditional_prior_dist ", len(conditional_prior_dist), conditional_prior_dist)
                    # conditional_prior_dist = np.ones(len(conditional_prior_dist))/len(conditional_prior_dist)
                    optimal_random_response_mechanism = Optimized_Randomized_Response(prior_dist = conditional_prior_dist, STATE_COUNT = state_count, INPUT_ALPHABET = convert_alphabet_to_string(individual_alphabet), normalized_objective_err_matrix = err_matrix, 
                        TOLERANCE_MARGIN = 0.01, APPROXIMATION = "LINEAR", solver = "SCS", is_kl_div = False, ALPHA=0.01, accelerate_from_rr=False)
                    mechanism_dict["0"] = optimal_random_response_mechanism
                    # sns.heatmap(optimal_random_response_mechanism.get_mechanism(2), annot=True)
                    # plt.show()
                    break
                __key = get_mechanism_key(predessor_index_list = predessor_index_list)
                # print(__key)
                if __key not in mechanism_dict.keys():
                    if uniform:
                        conditional_prior_dist = np.ones(len(individual_alphabet))/len(individual_alphabet)
                    else:
                        conditional_prior_dist = get_conditional_dist(i, predessor_index_list, (individual_alphabet), alphabet, prior_dist)
                    # err_matrix_2 = err_matrix*(1- conditional_prior_dist) # *(1/(conditional_prior_dist+0.00000001))
                    # err_matrix_2 = err_matrix_2 / np.max(err_matrix_2)
                    # sns.heatmap(err_matrix_2)
                    # plt.show()
                    print("conditional_prior_dist ", len(conditional_prior_dist), conditional_prior_dist)
                    # conditional_prior_dist = np.ones(len(conditional_prior_dist))/len(conditional_prior_dist)
                    optimal_random_response_mechanism = Optimized_Randomized_Response(prior_dist = conditional_prior_dist, STATE_COUNT = state_count, INPUT_ALPHABET = convert_alphabet_to_string(individual_alphabet), normalized_objective_err_matrix = err_matrix, 
                        TOLERANCE_MARGIN = 0.01, APPROXIMATION = "LINEAR", solver = "SCS", is_kl_div = True, ALPHA=0.01, accelerate_from_rr=False)
                    mechanism_dict[__key] = optimal_random_response_mechanism
                    # sns.heatmap(optimal_random_response_mechanism.get_mechanism(2), annot=True)
                    # plt.show()
            mechanism_list.append(mechanism_dict)

    def __get_mechanism_key__(self, predessor_index_list):
        __key = ""
        for i in predessor_index_list:
            __key += str(i) + " "
        return __key

    def __string_to_list__(self, s):
        l = []
        actual_split = s.split(" ")
        for j in actual_split:
            if j != "" :
                l.append((j))
        return l

    def __get_conditional_dist__(self, attribute_num, predessor_index_list, individual_alphabet, global_alphabet, prior_dist):
        __marginal_prob_dist = []
        k = len(predessor_index_list)
        # print("predessor_index_list ", predessor_index_list)
        for i in individual_alphabet:
            __marginal_prob = 0
            for index_j, j in enumerate(global_alphabet):
                # print("j[:k] ", j[:k], list(j[:k]) == list(predessor_index_list), j[attribute_num] == i, k == 0)
                if j[attribute_num] == i and (k == 0 or list(j[:k]) == list(predessor_index_list)):
                    __marginal_prob += prior_dist[index_j]
            __marginal_prob_dist.append(__marginal_prob)
        return __marginal_prob_dist/sum(__marginal_prob_dist)


    def __create_optimal_mechnism_dict__(self, alphabet = [], alphabet_dict = {}, prior_dist = [], err_type = "0_1", uniform = False):
        
            # print("mechanism_list ", mechanism_list)
        return mechanism_list

    def __get_randomized_value__(self, actual_value, eps, mechanism_list):
        actual_value = self.__string_to_list__(actual_value)
        last_value = None
        randomized_value_list = ""
        # print("actual_value ", actual_value)
        for index_i, i in enumerate(actual_value):
            if last_value == None:
                last_value = mechanism_list[index_i]["0"].gen_random_output(actual_value=i, eps=eps)[0]
            else:
                __key = randomized_value_list  #get_mechanism_key(predessor_index_list=randomized_value_list)
                last_value = mechanism_list[index_i][__key].gen_random_output(actual_value=i, eps=eps)[0]
            randomized_value_list += last_value + " "
        return randomized_value_list


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
    
    def gen_random_output(self, actual_value, eps, is_index = False):
        # print(f"{actual_value}  alphabet {self.INPUT_ALPHABET}")
        try:
            index_of_actual_value = self.INPUT_ALPHABET.index(actual_value)
        except ValueError:
            print(f"{actual_value} is not in the alphabet {self.INPUT_ALPHABET}")
            
        prob_vec = self.get_mechanism(eps)[index_of_actual_value,:]
        random_ = np.random.choice(len(self.INPUT_ALPHABET) if is_index else self.INPUT_ALPHABET, 1, p=prob_vec)
        return random_

    def get_name(self):
        return "Approximated Optimal Random Response"
