from label_encoder import *

class Dataset_handler():
    def __init__(self, data_dict, encodable_attributes = []):

        if not isinstance(data_dict, dict):
            raise TypeError(f"'data_dict' must be a dictionary. Found {type(data_dict)}")
        
        if not isinstance(encodable_attributes, list):
            raise TypeError(f"'encodable_attributes' must be a List. Found {type(encodable_attributes)}")
        
        self.__original_data_dict = data_dict
        self.__attributes = list(data_dict.keys())
        self.__encodable_labels = encodable_attributes
        self.__encoder_dict = {}

        for i in encodable_attributes:
            self.__encoder_dict[i] = Label_encoder(data_dict[i])

        self.__encoded_original_data_dict = self.encode_new_dataset(data_dict)
    
    def encode_new_dataset(self, dataset):
        if not isinstance(dataset, dict):
            raise TypeError(f"'data_dict' must be a dictionary. Found {type(dataset)}")
        
        columns = list(dataset.keys())
        self.validate_attributes(columns)

        for column in columns:
            if column in self.__encodable_labels:
                dataset[column] = self.__encoder_dict[column].encode_values(dataset[column])
        
        return dataset

    def decode_new_dataset(self, dataset):
        if not isinstance(dataset, dict):
            raise TypeError(f"'data_dict' must be a dictionary. Found {type(dataset)}")
        
        columns = list(dataset.keys())
        self.validate_attributes(columns)

        for column in columns:
            if column in self.__encodable_labels:
                dataset[column] = self.__encoder_dict[column].decode_values(dataset[column])
        
        return dataset
    
    def get_selected_attribute_data(self, required_attributes, is_encoded = True):
        seleceted_attribute_dataset = {}
        self.validate_attributes(required_attributes)

        for i in required_attributes:
            seleceted_attribute_dataset[i] = self.__encoded_original_data_dict[i]

    def get_orignal_dataset(self):
        return self.__original_data_dict
    
    def get_encoded_orignal_dataset(self):
        return self.__encoded_original_data_dict
    
    def get_attribute_list(self):
        return self.__attributes
    
    def validate_attributes(self, new_attributes):
        print("new_attributes", new_attributes)
        for i in new_attributes:
            if i not in self.__attributes:
                raise ValueError(f"Attribute {i} not in original dataset structure")
