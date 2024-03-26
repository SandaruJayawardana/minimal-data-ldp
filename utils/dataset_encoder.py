from label_encoder import *

class Dataset_encoder():
    def __init__(self, data, encodable_labels):
        self.encodable_labels = encodable_labels
        self.__encoder_dict = {}

        for i in encodable_labels:
            self.__encoder_dict[i] = Label_encoder(data[i].to_numpy())
    
    def encode_dataset(self, dataset):
        columns = dataset.columns.tolist()
        for column in columns:
            if column in self.encodable_labels:
                dataset[column] = self.__encoder_dict[self.__encoder_dict[column]].encode_values(dataset[column])
        
        return dataset

    def decode_dataset(self, dataset):
        columns = dataset.columns.tolist()
        for column in columns:
            if column in self.encodable_labels:
                dataset[column] = self.__encoder_dict[self.__encoder_dict[column]].decode_values(dataset[column])
        
        return dataset
    
# class Dataset_Encoder_New():
#     def __init__(self, dataset, alphabet_dict):
#         for index, row in dataset.iterrows():
#             for i in dataset.columns:


#     def get_encodered_dataset(self):
#         return self.encodered_dataset