import numpy as np

def create_alphabet(attributes_with_alphabet):
    if not(isinstance(attributes_with_alphabet, dict)):
        raise TypeError(f"attributes_with_alphabet should be a dictionary. Found {type(attributes_with_alphabet)}")
    
    attribute_list = list(attributes_with_alphabet.keys())
    attribute_count = len(attribute_list)
    num_propotions = 1

    remaining_alphabet = []
    for i in attribute_list:
        num_propotions *= len(attributes_with_alphabet[i])
        remaining_alphabet.append(attributes_with_alphabet[i])

    aggregated_alphabet = np.array(propotion_creator(selected_values = [], remaining_alphabet = remaining_alphabet))
    return aggregated_alphabet

def create_alphabet_ordered(attributes_with_alphabet, ordered_attribute_list):
    if not(isinstance(attributes_with_alphabet, dict)):
        raise TypeError(f"attributes_with_alphabet should be a dictionary. Found {type(attributes_with_alphabet)}")
    
    attribute_list = ordered_attribute_list # list(attributes_with_alphabet.keys())
    attribute_list.reverse()
    # print("revere", ordered_attribute_list, attribute_list)
    attribute_count = len(attribute_list)
    num_propotions = 1

    remaining_alphabet = []
    for i in attribute_list:
        num_propotions *= len(attributes_with_alphabet[i])
        remaining_alphabet.append(attributes_with_alphabet[i])

    aggregated_alphabet = np.array(propotion_creator(selected_values = [], remaining_alphabet = remaining_alphabet))
    return aggregated_alphabet

def propotion_creator(selected_values = [], remaining_alphabet = []):
    # print("selected_values", selected_values, "remaining_alphabet", remaining_alphabet)
    created_values = []
    if len(remaining_alphabet) == 0:
        raise ValueError("Alphabet size cannot be 0")
    
    if len(remaining_alphabet) == 1:
        # print("Leaf", remaining_alphabet)
        for i in remaining_alphabet[0]:
            created_values.append(selected_values + [i])
        # print("created_values ", created_values)
        return created_values
    
    for i in remaining_alphabet[0]:
        selected_values_copy = list(selected_values)
        selected_values_copy.append(i)
        created_values += (propotion_creator(selected_values_copy, remaining_alphabet[1:]))
    return created_values

def convert_alphabet_to_string(alphabet):
    converted_alphabet = []
    # print("a ", alphabet)
    # l, w = np.shape(alphabet)
    # print(l, w)
    for j in range(len(alphabet)):
        v = ""
        if isinstance(alphabet[0], list) or isinstance(alphabet[0], np.ndarray):
            for k in range(len(alphabet[0])):
                v += str(alphabet[j][k]) + " "
        else:
            v += str(alphabet[j])
        converted_alphabet.append(v)
    return converted_alphabet

def get_alphabet_dict_from_data(data):
    attributes = list(data.columns)
    alphabet_dict = {}
    for i in attributes:
        alphabet = np.unique(data[i])
        alphabet_dict[i] = list(alphabet)
    return alphabet_dict

# print(propotion_creator([], [[1, 2, 3], [4, 5], [5, 6, 7 , 8, 9]]))