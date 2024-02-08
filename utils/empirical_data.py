import numpy as np

def joint_prob_binary(xn, yn):
    if type(xn) == list:
        xn = np.array(xn)
    if type(yn) == list:
        yn = np.array(yn)

    if xn.ndim == 1:
        xn = np.reshape(xn,(len(xn), 1))
    if yn.ndim == 1:
        yn = np.reshape(yn,(len(yn), 1))

    symbols_xn = np.sort(np.unique(xn, axis=0))
    symbols_yn = np.sort(np.unique(yn, axis=0))
    xn = np.concatenate((xn,yn), axis=1)
    [symbols, counts] = np.unique(xn, axis=0, return_counts=True)

    # print(symbols, counts)

    tot_count = len(xn)
    normalized_count_dict = dict({})

    for i, joint_symb in enumerate(symbols):
        normalized_count_dict[str(joint_symb)] = counts[i]/tot_count

    # print(normalized_count_dict)
    joint_prob_matrix = np.zeros((len(symbols_xn), len(symbols_yn)))
    # print(np.shape(joint_prob_matrix))
    # print(symbols_xn)
    matrix_symb = []
    for yn_symb in symbols_yn:
        for xn_symb in symbols_xn:
            key_ = str(xn_symb)[:-1] + " "+ str(yn_symb)[1:]
            matrix_symb.append(key_)

    for i, xn_symb in enumerate(symbols_xn):
        for j, yn_symb in enumerate(symbols_yn):
            key_ = str(xn_symb)[:-1] + " "+ str(yn_symb)[1:]
            # print(key_)
            if key_ in normalized_count_dict.keys():
                joint_prob_matrix[j][i] = normalized_count_dict[key_]

    return matrix_symb, symbols_xn, symbols_yn, joint_prob_matrix

# print(joint_prob_binary([1,1,1,0,0], [1,1,1,0,0]))          
# print(joint_prob_binary(["5", "3", "1", "1", "1", "0", "0", "1"], ["5", "3", "1", "1", "1", "0", "0", "0"]))
