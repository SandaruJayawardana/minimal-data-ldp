import numpy as np

def error_cal(actual, perturbed, alphabet_dict={}, err_type = "0_1", check = False):
    if check and isinstance(actual, str):
        actual = list(alphabet_dict.keys()).index(actual)
        perturbed = list(alphabet_dict.keys()).index(perturbed)
    # print(actual, perturbed)
    if err_type == "0_1":
        # printactual, perturbed)
        return 0 if (str(actual) == str(perturbed)) else 1
    elif err_type == "l1":
        return np.linalg.norm(((actual)-(perturbed)), 1) # abs((actual)-(perturbed)) # np.linalg.norm(((actual)-(perturbed)), 1)
    elif err_type == "l2":
        return np.linalg.norm(((actual)-(perturbed)), 2) # ((actual)-(perturbed))**2 # np.linalg.norm(((actual)-(perturbed)), 2)
    else:
        raise TypeError(f"Unknown error type {err_type}")

class Normalize_error_matrix:
    def __init__(self, attribute_list, alphabet, priority_dict, alphabet_dict = {}, err_type = "0_1"):
        # print(alphabet)
        self.attribute_list = attribute_list
        self.alphabet = alphabet
        self.priority_dict = priority_dict
        self.alphabet_dict = alphabet_dict
        self.err_type = err_type

        self.normalized_error_matrix = np.zeros((len(alphabet), len(alphabet)))
        # print(np.shape(self.normalized_error_matrix))

        for index_i, i in enumerate(alphabet): # input
            for index_j, j in enumerate(alphabet): # output
                # print("i", i, "j", j)
                # print(self.normalized_error_matrix[i][j])
                # print(index_i, index_j)
                # print(i, j)
                err = self.err(i, j)
                self.normalized_error_matrix[index_i][index_j] = err
        self.__max_value = np.max(self.normalized_error_matrix)
        self.normalized_error_matrix /= self.__max_value

    # Original
    # def err(self, actual, perturbed):
    #     # print("actual", actual, "perturbed", perturbed)
    #     err = 0
    #     for k in range(len(self.alphabet[0])):
    #         attribute = self.attribute_list[k]
    #         # print("attribute", attribute)
    #         if attribute in list(self.priority_dict.keys()):
    #             priority = self.priority_dict[attribute]
    #         else:
    #             priority = 1
    #         # print("priority", priority)
    #         err += error_cal(actual=actual[k], perturbed=perturbed[k], err_type="0_1")*priority
    #         # print("error_cal", error_cal(actual=actual[k], perturbed=perturbed[k], err_type="0_1")*priority)
    #     # print("err", err)
    #     return err
    
    def err(self, actual, perturbed):
        # print("actual", actual, "perturbed", perturbed)
        actual_arr = np.zeros(len(self.alphabet[0]), dtype=np.int64)
        perturb_arr = np.zeros(len(self.alphabet[0]), dtype=np.int64)
        err = 0
        for k in range(len(self.alphabet[0])):
            attribute = self.attribute_list[k]
            # print("attribute", attribute)
            if attribute in list(self.priority_dict.keys()):
                priority = self.priority_dict[attribute]
            else:
                priority = 1
            # print("priority", priority)
            actual_arr[k] = int(actual[k])*priority
            perturb_arr[k] = int(perturbed[k])*priority
            # err += error_cal(actual=actual[k], perturbed=perturbed[k], err_type="l1")*priority
            # print("error_cal", error_cal(actual=actual[k], perturbed=perturbed[k], err_type="0_1")*priority)
        
        # print("actual_arr ", actual_arr)
        # print("perturb_arr ", perturb_arr)
        # err = error_cal(actual=self.alphabet_dict[str(actual_arr)], perturbed=self.alphabet_dict[str(perturb_arr)], err_type="0_1")
        err = error_cal(actual=actual_arr, perturbed=perturb_arr, err_type=self.err_type)
        # print("err", err)
        return err

    def get_value_error(self, actual, perturbed):
        # print(actual, perturbed)
        if isinstance(actual, str):
            actual_split = actual.split(" ")
            actual = []
            # print(actual_split)
            for i in actual_split:
                # print(i)
                if i != "" :
                    actual.append(int(i))
            actual = np.array(actual)

            perturbed_split = perturbed.split(" ")
            perturbed = []
            for i in perturbed_split:
                if i != "":
                    perturbed.append(int(i))
            perturbed = np.array(perturbed)
        # print(actual, perturbed)
        err = self.err(actual=actual, perturbed=perturbed)
        return err/self.__max_value
