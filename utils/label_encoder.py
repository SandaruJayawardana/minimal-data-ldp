import numpy as np

class Label_encoder():
    def __init__(self, data):
        unique_elements = np.unique(data)
        self.__encoding_dict = {}
        self.__decoding_dict = {}

        for i, elem in enumerate(unique_elements):
            self.__encoding_dict[elem] = i
            self.__encoding_dict[i] = elem
    
    def encode_value(self, value):
        return self.__encoding_dict[value]
    
    def encode_values(self, values):
        encoded_values = []
        for value in values:
            encoded_values.append(self.__encoding_dict[value])
        
        return encoded_values
    
    def decode_value(self, value):
        return self.__decoding_dict[value]
    
    def decode_values(self, values):
        decoded_values = []
        for value in values:
            decoded_values.append(self.__decoding_dict[value])
        
        return decoded_values