# '''
#     Utility functions
# '''
# import numpy as np

# def validate_err_matrix(err_matrix, dim):

#     if not(isinstance(err_matrix, np.ndarray) and err_matrix.ndim == 2):
#         assert("Error matrix is not valid numpy array or not a 2 dim matrix")
#     elif not(err_matrix.shape[0] == dim and err_matrix.shape[1] == dim):
#         assert(f"Error matrix is not valid. Matrix dim not match. Dim 1 has {err_matrix.shape[0]}, Dim 2 has {err_matrix.shape[1]}, expected {dim}")
#     # print(err_matrix, dim)
#     if np.all(np.diag(err_matrix) == np.zeros(dim)):
#         assert(f"Digonal needs to be zero. But found {np.diag(err_matrix)}")

#     if np.max(err_matrix) > 1:
#         assert(f"Matrix should be between 0 and 1. But found {np.max(err_matrix)}")

#     if np.min(err_matrix) > 1:
#         assert(f"Matrix should be between 0 and 1. But found {np.min(err_matrix)}")
    
#     return True

# def validate_distribution(prior_dist, n, terminate = True):
#     # print("ut", prior_dist, np.min(prior_dist))
#     # assert isinstance(prior_dist, np.ndarray), "Not a Numpy array"
#     if terminate:
#         assert abs(np.sum(prior_dist) - 1) < 0.1, f"Prior distribution should be sum to 1. Found {np.sum(prior_dist):.6f}"
#         assert not(np.min(prior_dist) < 0 or np.max(prior_dist) > 1), f"Probabilities should be between 0 and 1. Found {np.min(prior_dist) if np.min(prior_dist) < 0 else np.max(prior_dist)}"
#         assert not(len(prior_dist) != n), "Prior distribution and state count do not match"
    
#     if not(terminate): 
#         return (abs(np.sum(prior_dist) - 1) < 0.1) and (not(np.min(prior_dist) < 0 or np.max(prior_dist) > 1)) and (not(len(prior_dist) != n))
    
#     return True

# # def validate_distribution(prior_dist):
# #     if isinstance(prior_dist, np.ndarray) and abs(np.sum(prior_dist) - 1) < 0.00001:
# #         assert(f"Prior distribution should be sum to 1. Found {np.sum(prior_dist)}")
# #     return True

# def validate_alphabet(alphabet):
#     if len(alphabet) != len(set(alphabet)):
#         assert("Alphabet should contain unique elements")
#     return True

# def cal_tv(p, q):
#     sum_ = np.sum(np.abs(p - q))
#     return sum_/2

# def divide_range_into_slices(start, end, num_slices):
#     slice_size = (end - start) / num_slices
#     slices = [start + i*slice_size for i in range(num_slices)]
#     slices.append(end)
#     return slices

'''
    Utility functions
'''
import numpy as np

def validate_err_matrix(err_matrix, dim):

    if not(isinstance(err_matrix, np.ndarray) and err_matrix.ndim == 2):
        assert False, "Error matrix is not valid numpy array or not a 2 dim matrix"
    elif not(err_matrix.shape[0] == dim and err_matrix.shape[1] == dim):
        assert False, f"Error matrix is not valid. Matrix dim not match. Dim 1 has {err_matrix.shape[0]}, Dim 2 has {err_matrix.shape[1]}, expected {dim}"
    # print(err_matrix, dim)
    if np.all(np.diag(err_matrix) != np.zeros(dim)):
        assert False, f"Digonal needs to be zero. But found {np.diag(err_matrix)}"

    if np.max(err_matrix) > 1:
        assert False, f"Matrix should be between 0 and 1. But found {np.max(err_matrix)}"

    if np.min(err_matrix) < 0:
        assert False, f"Matrix should be between 0 and 1. But found {np.min(err_matrix)}"
    
    return True

def validate_distribution(prior_dist, n, terminate = True):
    # print("ut", prior_dist, np.min(prior_dist))
    # assert isinstance(prior_dist, np.ndarray), "Not a Numpy array"
    if terminate:
        assert abs(np.sum(prior_dist) - 1) < 0.1, f"Prior distribution should be sum to 1. Found {np.sum(prior_dist):.6f}"
        assert not(np.min(prior_dist) < 0 or np.max(prior_dist) > 1), f"Probabilities should be between 0 and 1. Found {np.min(prior_dist) if np.min(prior_dist) < 0 else np.max(prior_dist)}"
        assert not(len(prior_dist) != n), "Prior distribution and state count do not match"
    
    if not(terminate): 
        return (abs(np.sum(prior_dist) - 1) < 0.1) and (not(np.min(prior_dist) < 0 or np.max(prior_dist) > 1)) and (not(len(prior_dist) != n))
    
    return True

# def validate_distribution(prior_dist):
#     if isinstance(prior_dist, np.ndarray) and abs(np.sum(prior_dist) - 1) < 0.00001:
#         assert(f"Prior distribution should be sum to 1. Found {np.sum(prior_dist)}")
#     return True

def validate_alphabet(alphabet):
    if len(alphabet) != len(set(alphabet)):
        assert False, "Alphabet should contain unique elements"
    return True

def cal_tv(p, q):
    sum_ = np.sum(np.abs(p - q))
    return sum_/2

def divide_range_into_slices(start, end, num_slices):
    slice_size = (end - start) / num_slices
    slices = [start + i*slice_size for i in range(num_slices)]
    slices.append(end)
    return slices
