import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from utils.alphabet import *

from utils.empirical_data import *
from mechanisms.mldp.mldp import *
from mechanisms.k_rr.k_rr_mechanism import *
from mechanisms.mldp.repetitive_optimizer import *
# from utils.synthetic_dataset import *
from mechanisms.exponential.exponential_mechanism import *
from utils.simpleinfotheory import *
from utils.normalize_error_matrix import *
from mechanisms.rappor.rappor_mechanism import *

import matplotlib as mpl
import matplotlib.lines as mlines


def list_to_string(l):
    s = ""
    for i in l:
        s += str(i) + " "
    return s

def string_to_list(s):
    l = []
    actual_split = s.split(" ")
    for j in actual_split:
        if j != "" :
            l.append((j))
    return l

def get_key(parent_nodes, __ordered_index_dict, randomized_value_list):
    # print("parent_nodes ", parent_nodes, "__ordered_index_dict ", __ordered_index_dict, "randomized_value_list ", randomized_value_list)
    __key = ""
    if len(parent_nodes) > 1:
        for i in parent_nodes[:-1]:
            # print("randomized_value_list ", randomized_value_list, "__ordered_index_dict[i]", __ordered_index_dict[i])
            __key += randomized_value_list[__ordered_index_dict[i]] + " "
    return __key

class Revised_Mechanism(Privacy_Mechanism):

    def __init__(self, data, model, ordered_attribute_list, parent_node_dict, alphabet_dict, priority_dict, uniform = False, err_type = "0_1", TOLERANCE_MARGIN = 0.01, APPROXIMATION = "LINEAR", solver = "SCS", is_kl_div = True, ALPHA=0.01, accelerate_from_rr = False):
        self.data = data
        self.model = model
        self.ordered_attribute_list = ordered_attribute_list
        self.priority_dict = priority_dict
        self.parent_node_dict = parent_node_dict
        self.alphabet_dict = alphabet_dict
        self.mechanism_list = []
        self.uniform = uniform
        self.err_type = err_type
        self.__eps = 0
        self.__ordered_index_dict = {}
        self.__ordered_index_list = []
        for index_i, i in enumerate(self.ordered_attribute_list):
            self.__ordered_index_dict[i] = index_i
            self.__ordered_index_list.append(list(data.columns).index(i))

    def get_conditional_dist(self, attributes_list, joint_probability_dict, individual_alphabet, parent_nodes_values):
        __marginal_prob_dist = []
        k = len(attributes_list) - 1
        # print("joint_probability_dict ", joint_probability_dict)
        for i in individual_alphabet:
            __marginal_prob = 0
            for index_j, j_ in enumerate(joint_probability_dict.keys()):
                j = string_to_list(j_)
                # print("j", j, j[-1], i, j[-1]==i)
                if j[-1] == i and (k == 0 or list(j[:k]) == list(parent_nodes_values)):
                    __marginal_prob += joint_probability_dict[j_]
            __marginal_prob_dist.append(__marginal_prob)
        # print(__marginal_prob_dist)
        return np.array(__marginal_prob_dist)/sum(__marginal_prob_dist)
    
    def get_local_alphabet(self, attributes_list):
        local_alphabet_dict = {}

        for i in attributes_list:
            local_alphabet_dict[i] = self.alphabet_dict[i]

        return create_alphabet(attributes_with_alphabet=local_alphabet_dict)
    
    def cal_joint_probability(self, local_alphabet, dataset):
        joint_probability_dict = {}
        # print("local_alphabet ", local_alphabet)
        for i in local_alphabet:
            joint_probability_dict[list_to_string(i)] = 3
        # print("joint_probability_dict ", joint_probability_dict)
        # print(dataset)
        for index, row in dataset.iterrows():
            row_as_list = row.tolist()
            # print("list_to_string(row_as_list) ", list_to_string(row_as_list))
            # print(row_as_list)
            joint_probability_dict[list_to_string(row_as_list)] += 1
        
        data_count = len(dataset)
        for i in local_alphabet:
            joint_probability_dict[list_to_string(i)] /= data_count
        
        return joint_probability_dict

    def create_optimal_mechanism_list(self):
        for i in self.ordered_attribute_list:
            if not(isinstance(i, str)):
                attribute = str(i)
            else:
                attribute = i

            mechanism_dict = {}
            individual_alphabet = self.alphabet_dict[attribute]
            state_count = len(individual_alphabet)
            # print("alp", type(individual_alphabet), individual_alphabet)
            normalize_error_matrix = Normalize_error_matrix(attribute_list=[attribute], alphabet=individual_alphabet, priority_dict=self.priority_dict, alphabet_dict=self.alphabet_dict, err_type=self.err_type)
            err_matrix = normalize_error_matrix.normalized_error_matrix
            sns.heatmap(err_matrix)
            plt.show()

            attributes_list = self.parent_node_dict[attribute]
            if len(attributes_list) == 0:
                attributes_list = [attribute]
            else:
                attributes_list.append(attribute)
            # print("attributes_list ", attributes_list)
            dataset = self.data[attributes_list]
            local_alphabet = self.get_local_alphabet(attributes_list=attributes_list)
            joint_probability = self.cal_joint_probability(local_alphabet=local_alphabet, dataset=dataset)

            for index_j, j in enumerate(local_alphabet):
                parent_nodes_values = j[:-1]
                if len(parent_nodes_values) == 0:
                    if self.uniform:
                        conditional_prior_dist = np.ones(len(individual_alphabet))/len(individual_alphabet)
                    else:
                        conditional_prior_dist = self.get_conditional_dist(attributes_list=attributes_list, joint_probability_dict=joint_probability, individual_alphabet=individual_alphabet, parent_nodes_values=parent_nodes_values)

                    optimal_random_response_mechanism = Optimized_Randomized_Response(prior_dist = conditional_prior_dist, STATE_COUNT = state_count, INPUT_ALPHABET = convert_alphabet_to_string(individual_alphabet), normalized_objective_err_matrix = err_matrix, 
                        TOLERANCE_MARGIN = 0.01, APPROXIMATION = "LINEAR", solver = "SCS", is_kl_div = False, ALPHA=0.01, accelerate_from_rr=False)
                    mechanism_dict["0"] = optimal_random_response_mechanism
                    break
                __key = list_to_string(parent_nodes_values)
                # print(__key)
                if __key not in mechanism_dict.keys():
                    if self.uniform:
                        conditional_prior_dist = np.ones(len(individual_alphabet))/len(individual_alphabet)
                    else:
                        conditional_prior_dist = self.get_conditional_dist(attributes_list=attributes_list, joint_probability_dict=joint_probability, individual_alphabet=individual_alphabet, parent_nodes_values=parent_nodes_values)
                    optimal_random_response_mechanism = Optimized_Randomized_Response(prior_dist = conditional_prior_dist, STATE_COUNT = state_count, INPUT_ALPHABET = convert_alphabet_to_string(individual_alphabet), normalized_objective_err_matrix = err_matrix, 
                        TOLERANCE_MARGIN = 0.01, APPROXIMATION = "LINEAR", solver = "SCS", is_kl_div = True, ALPHA=0.01, accelerate_from_rr=False)
                    mechanism_dict[__key] = optimal_random_response_mechanism
            self.mechanism_list.append(mechanism_dict)
        return self.mechanism_list

    def get_mechanism(self):
        if self.mechanism_list == []:
            self.create_optimal_mechanism_list()
        return self.mechanism_list

    def get_eps(self):
        return self.__eps
    
    def gen_random_output(self, actual_value, eps):
        actual_value = string_to_list(actual_value)
        last_value = None
        randomized_value_list = []
        randomized_value_list_final = np.zeros(len(self.ordered_attribute_list)).tolist()
        self.__eps = eps

        if not(isinstance(eps, list)):
            eps = [eps] * len(self.ordered_attribute_list)
        # print("eps list ", eps)
        # print("actual_value ", actual_value)
        # print("self.ordered_attribute_list", self.ordered_attribute_list)
        # print("self.__ordered_index_list", self.__ordered_index_list)
        # print("self.parent_node_dict", self.parent_node_dict)
        # print("randomized_value_list_final", randomized_value_list_final)
        for index_i, i_ in enumerate(self.__ordered_index_list):
            
            i = actual_value[i_]
            # print("self.ordered_attribute_list", self.ordered_attribute_list, "self.ordered_attribute_list[index_i]", self.ordered_attribute_list[index_i], "self.parent_node_dict", self.parent_node_dict, "self.parent_node_dict[self.ordered_attribute_list[index_i]]", self.parent_node_dict[self.ordered_attribute_list[index_i]])
            if len(self.parent_node_dict[self.ordered_attribute_list[index_i]]) == 0:
                last_value = self.mechanism_list[index_i]["0"].gen_random_output(actual_value=i, eps=eps[index_i])[0]
            else:
                # print("self.mechanism_list[index_i]", self.mechanism_list[index_i])
                # print("index_i ", index_i)
                __key = get_key(parent_nodes = self.parent_node_dict[self.ordered_attribute_list[index_i]], __ordered_index_dict=self.__ordered_index_dict, randomized_value_list=randomized_value_list)  #get_mechanism_key(predessor_index_list=randomized_value_list)
                
                # print("__key", __key)
                last_value = self.mechanism_list[index_i][__key].gen_random_output(actual_value=i, eps=eps[index_i])[0]
            randomized_value_list.append(last_value)
            randomized_value_list_final[i_] = last_value
        #     print("i_", i_, "i", i, "random ", last_value)
        # print("randomized_value_list ", randomized_value_list, "randomized_value_list_final", randomized_value_list_final)
        return randomized_value_list_final
    
    def get_expected_utility_error(self, eps, input_probability = None):
        return np.sum(input_probability * np.sum(self.get_mechanism(eps) * self.normalized_objective_err_matrix, axis= 1))

    def get_name(self):
        return "Revised Optimal RR"
    