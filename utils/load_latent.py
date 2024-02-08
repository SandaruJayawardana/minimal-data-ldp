"""
Task: 
Author: Sandaru Jayawardana
"""
import numpy as np
import os
import pickle

FILE_LOCATION = "/Users/sandarujayawardana/Downloads"

'''
Return - trainx, trainy, testx, testy
'''

def pre_process_celebA(latent_file_name = "celebA_latents/latent.npy", detail_file = "celebA_latents/indexes", 
                       attr_list_file = "celeba_anno/list_attr_celeba.txt", TRAINING_AMOUNT = 0.7):

    # Read latents
    latents = np.load(FILE_LOCATION + "/" + latent_file_name)

    # Read index file
    indexes = pickle.load(open(FILE_LOCATION + "/" + detail_file, "rb"))
    
    # Read celebA attr txt
    attr_labels = []

    # Reading txt
    with open(FILE_LOCATION + "/" + attr_list_file, 'r') as txt_file:
        no_of_lines = int(txt_file.readline())
        fields = txt_file.readline()

        for i in range(no_of_lines):
            rows = txt_file.readline().split()
            attr_labels.append(rows)

    trainx = [] #np.array((int(int(no_of_lines) * TRAINING_AMOUNT), 512)) # latents
    trainy = [] #np.array((int(int(no_of_lines) * TRAINING_AMOUNT), 40)) # 40 binary attributes

    testx = [] #np.array((int(no_of_lines) - int(int(no_of_lines) * TRAINING_AMOUNT), 512))
    testy = [] #np.array((int(no_of_lines) - int(int(no_of_lines) * TRAINING_AMOUNT), 40))

    for i, img_index in enumerate(indexes):
        img_index_ = int(str(img_index)[:-4]) - 1
        # print(img_index, img_index_)
        if img_index_ < int(no_of_lines * TRAINING_AMOUNT):
            trainx.append(latents[i])
            trainy.append(attr_labels[img_index_][1:])
            # print(attr_labels[img_index_][0], img_index)
        else:
            testx.append(latents[i])
            testy.append(attr_labels[img_index_][1:])
            # print(attr_labels[img_index_][0], img_index)

    return np.array(trainx, dtype = np.float64), np.array(trainy, dtype = np.float64), np.array(testx, dtype = np.float64), np.array(testy, dtype = np.float64)


# a,b,c,d = pre_process_celebA()

# print(np.shape(a), np.shape(b), np.shape(c), np.shape(d))
