import numpy as np

class Prior_distribution_calc():
    def __init__(self, attribute_list, alphabet, data):
        DEFAULT_VALUE = 10

        sample_count = len(data[attribute_list[0]])
        hist_dict = {}
        count = 0

        for i in alphabet:
            v = ""
            for j in i:
                v += str(j) + " "
            hist_dict[v] = DEFAULT_VALUE #default value
        for i in range(sample_count):
            v = ""
            for attr in attribute_list:
                v += str(data[attr][i]) + " "
            hist_dict[v] += 1
        

        count = DEFAULT_VALUE * len(alphabet) + sample_count

        for key_ in list(hist_dict.keys()):
            hist_dict[key_] /= count
        
        self.__prior_distribution = []

        for i in alphabet:
            v = ""
            for j in i:
                v += str(j) + " "
            self.__prior_distribution.append(hist_dict[v])

    def get_prior_distribution(self):
        return self.__prior_distribution
    