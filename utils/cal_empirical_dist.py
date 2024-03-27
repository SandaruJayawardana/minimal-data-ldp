

class Calculate_Empirical_Distribution():
    def __init__(self, alphabet, dataset, default_value = 3):
        joint_probability_dict = {}
        # print("local_alphabet ", local_alphabet)
        for i in alphabet:
            joint_probability_dict[self.list_to_string(i)] = default_value
        # print("joint_probability_dict ", joint_probability_dict)
        # print(dataset)
        for index, row in dataset.iterrows():
            row_as_list = row.tolist()
            # print("list_to_string(row_as_list) ", list_to_string(row_as_list))
            # print(row_as_list)
            joint_probability_dict[self.list_to_string(row_as_list)] += 1
        
        data_count = len(dataset)
        for i in alphabet:
            self.joint_probability_dict[self.list_to_string(i)] /= data_count
     
    def get_probability_dist(self):
        return self.joint_probability_dict
    
    def list_to_string(self, l):
        s = ""
        for i in l:
            s += str(i) + " "
        return s
