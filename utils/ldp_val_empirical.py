import pandas as pd
import numpy as np

def list_to_string(l):
    s = ""
    for i in l:
        s += str(i) + " "
    return s

class LDP_Empirical_Validator():
    def __init__(self, mechanism, dataset, alphabet_dict, alphabet_list, repetion_count = 10):
        if not(isinstance(dataset, pd.DataFrame)):
            raise ValueError("Dataset is not a Pandas Dataframe!")
        NUM_ATTRIBUTES = len(dataset.columns)
        DIM = 1

        self.mechanism = mechanism
        self.dataset = dataset
        self.alphabet_dict = alphabet_dict
        self.repetion_count = repetion_count

        for i in list(alphabet_dict.values()):
            print(len(i))
            DIM *= len(i)
        # print("DIM ", DIM)
        self.base_values = np.ones((NUM_ATTRIBUTES))

        previous_factor = 1
        
        for index_i, i in enumerate(self.dataset.columns):
            
            if index_i == 0:
                prev_attribute = i
                continue
            self.base_values[index_i] = len(self.alphabet_dict[prev_attribute])*previous_factor
            previous_factor = len(self.alphabet_dict[prev_attribute])*previous_factor
            prev_attribute = i
        # print("self.base_values ", self.base_values)
            
        self.individual_x_index_of_alphabet_dict = {}
        # print("dataset.columns", dataset.columns)
        # print("alphabet_list", alphabet_list)

        
        for __attribute_num, __attribute in enumerate(dataset.columns):
            self.individual_x_index_of_alphabet_dict[__attribute] = {}
            for __alphabet_val in alphabet_dict[__attribute]:
                self.individual_x_index_of_alphabet_dict[__attribute][__alphabet_val] = []
                for index_i, i in enumerate(alphabet_list):
                    # print(i[NUM_ATTRIBUTES - 1 - __attribute_num], __alphabet_val)
                    # if i[NUM_ATTRIBUTES - 1 - __attribute_num] == __alphabet_val:
                    if i[__attribute_num] == __alphabet_val:
                        # print("Trig")
                        self.individual_x_index_of_alphabet_dict[__attribute][__alphabet_val].append(index_i)

        print("self.individual_x_index_of_alphabet_dict", self.individual_x_index_of_alphabet_dict)
        # raise ValueError("AAAA")
        self.results_matrix = np.zeros((DIM, DIM)) # {key: randomized output. value: dictionary {key: list of input values, value: num of occurance}

    def __get_index__(self, x):
        index_of_x_i = 0
        for index_i, i in enumerate(self.dataset.columns):
            index_of_x_i += self.alphabet_dict[i].index(x[index_i])*self.base_values[index_i]

        return index_of_x_i

    def run_test(self, eps):
        for index, row in self.dataset.iterrows():
            row_as_list = row.tolist()
            # print("eeee")
            actual_value = list_to_string(row_as_list)
            # print(index)
            # if index > 10000:
            #     break
            x_index = int(self.__get_index__(row))
            for i in range(self.repetion_count):
                # print("actual_value ", actual_value)
                y_index = int(self.__get_index__(self.mechanism.gen_random_output(actual_value, eps)))
                # print(x_index, y_index)
                self.results_matrix[x_index, y_index] += 1
        
        return (self.results_matrix)
        
