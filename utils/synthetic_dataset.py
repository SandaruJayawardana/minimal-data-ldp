import numpy as np
from utils.util_functions import *
from utils.divergence import *
from utils.simpleinfotheory import mutualinformation, entropy

class Gen_Synthetic_Dataset:
    def __init__(self, no_of_states = 4, no_of_samples = 1000, alphabet = []):
        self.NO_OF_STATES = no_of_states
        self.SAMPLE_COUNT = no_of_samples
        self.__uniform_data__ = None
        self.__random_data__ = None
        self.__custom_data__ = None
        # self.alphabet = np.arange(no_of_states)
        self.__distribution = None
        self.alphabet = alphabet

    def __gen_synthetic__(self, distribution):
        synthetic_dataset = []
        for i in range(self.SAMPLE_COUNT):
            synthetic_dataset.append(np.random.choice(self.alphabet, 1, p=distribution))
        return [i[0] for i in synthetic_dataset]
    
    def get_distribution(self):
        if self.__distribution == None:
            assert("Distribution is not defined")
        return self.__distribution
    
    def get_uniform_data_sample(self):
        if self.__uniform_data__ == None:
            self.__uniform_data__ = self.gen_uniform()
        return self.__uniform_data__

    def gen_uniform(self):
        self.__uniform_data__ = self.__gen_synthetic__(np.ones(self.NO_OF_STATES)/self.NO_OF_STATES)
        return self.__uniform_data__
    
    def get_custom_data_sample(self, distribution):
        if self.__custom_data__ == None:
            self.__custom_data__ = self.__gen_synthetic__(distribution)
        return self.__custom_data__
    
    def gen_custom(self, distribution):
        self.__custom_data__ = self.__gen_synthetic__(distribution)
        return self.__custom_data__
    
    def gen_random(self):
        raise ValueError("Not Implemented")
        # self.__uniform_data__ = self.__gen_synthetic__(np.random())
        # return self.__uniform_data__

class Gen_Synthetic_Distribution:
    

    def __init__(self, original_dist = [0.5, 0.5], div_type = "TV", no_samples = 10, sample_count_per_sample = 100, attribute_state_count = []):
        # Parameters
        self.CONST_ITERATION_FACTOR = 200
        self.ROUNDING_PRECISION = 1
        self.attribute_state_count = attribute_state_count

        validate_distribution(original_dist, len(original_dist))
        self.__ORIGINAL_DIST = original_dist
        self.__NO_SAMPLES = no_samples
        self.__SAMPLE_COUNT_PER_SAMPLE = sample_count_per_sample
        self.DIV_TYPE = div_type

        if div_type == "MI":
            # print(self.cal_mutual_info(np.ones(len(original_dist))/len(original_dist)))
            self.__divergence_dist_list = divide_range_into_slices(start=0, end=entropy(np.ones(len(original_dist))/len(original_dist)), num_slices=self.__NO_SAMPLES)
        else:
            self.__divergence_dist_list = divide_range_into_slices(start=0, end=1, num_slices=self.__NO_SAMPLES)
        # print(self.__divergence_dist_list)
        self.__is_created_dist = False
        self.__synthetic_dist = {}
        for i in self.__divergence_dist_list:
            self.__synthetic_dist[round(i, self.ROUNDING_PRECISION)] = []
        self.__synthetic_dist[round(self.__divergence_dist_list[0], self.ROUNDING_PRECISION)] = [original_dist]

    def get_synthetic_distribution(self):
        # self. create_distribution()
        assert self.__is_created_dist, "Create distribuiton first"
        return self.__synthetic_dist

    def create_distribution(self):
        self.__is_created_dist = True
        MAX_ITERATIONS = self.CONST_ITERATION_FACTOR * (self.__NO_SAMPLES * self.__SAMPLE_COUNT_PER_SAMPLE)*10
        cal_div = Divergence(self.DIV_TYPE)

        for i in range(MAX_ITERATIONS):
            new_dist = np.abs(self.__ORIGINAL_DIST + np.random.normal(0, 1+i/20, len(self.__ORIGINAL_DIST)))
            new_dist = new_dist/np.sum(new_dist)
            # print(new_dist)
            if self.DIV_TYPE == "MI":
                divergence_ = round(self.cal_mutual_info(new_dist), self.ROUNDING_PRECISION)
            else:
                divergence_ = round(cal_div.cal_divergence(self.__ORIGINAL_DIST, new_dist), self.ROUNDING_PRECISION)
            # print(divergence_)
            if divergence_ in self.__synthetic_dist.keys():
                if len(self.__synthetic_dist[divergence_]) < self.__SAMPLE_COUNT_PER_SAMPLE:
                    self.__synthetic_dist[divergence_].append(new_dist)

    def interpolate(self, a, b, alpha):
        # a,b = b,a
        return alpha * a + (1-alpha) * b

    def cal_mutual_info(self, p):
        # print(self.attribute_state_count)
        joint_p = np.zeros((self.attribute_state_count[0], self.attribute_state_count[1]))
        width_ = (self.attribute_state_count[1])
        for i in range(self.attribute_state_count[0]):
            joint_p[i,:] = p[i*width_:(i+1)*width_]
        
        return mutualinformation(joint_p)

    def create_distribution_away(self):
        self.__is_created_dist = True
        uniform_dist = np.ones(len(self.__ORIGINAL_DIST))/(len(self.__ORIGINAL_DIST))
        N = 10
        STEP = 0.1 #np.min(np.abs(uniform_dist - self.__ORIGINAL_DIST))/N
        alpha = 1 # np.ones(len(self.__ORIGINAL_DIST))
        self.__synthetic_dist = []

        

        for i in range(N):
            alpha += STEP
            new_dist = self.interpolate(a=self.__ORIGINAL_DIST, b=uniform_dist, alpha=alpha)
            print(new_dist)
            if validate_distribution(new_dist, len(new_dist), terminate=False):
                self.__synthetic_dist.append(new_dist)
            else:
                break