import numpy as np
from mechanisms.privacy_mechanism import *
from utils.util_functions import *
import matplotlib.pyplot as plt
# import seaborn as sns


class Rappor_mechanism(Privacy_Mechanism):

    def __init__(self, STATE_COUNT, INPUT_ALPHABET = [], prob_f=0.5, prob_p=0.5, prob_q=0.75, collection_count = 5):
        self.STATE_COUNT = STATE_COUNT
        self.collection_count = collection_count

        if len(INPUT_ALPHABET) != STATE_COUNT:
            self.INPUT_ALPHABET = str(range(STATE_COUNT))
        elif validate_alphabet(INPUT_ALPHABET):
            self.INPUT_ALPHABET = INPUT_ALPHABET

        self.prob_f = prob_f  # Probability of adding noise
        self.prob_p = prob_p  # Probability of a bit being 1
        self.prob_q = prob_q  # Probability of a bit remaining 1

    
    def _apply_permanent_randomized_response(self, value):
        """
        Apply the first layer of RAPPOR, Permanent Randomized Response (PRR).
        """
        prr = np.random.choice([0, 1, value], p=[self.prob_f / 2, self.prob_f / 2, 1 - self.prob_f])
        return prr

    def _apply_instantaneous_randomized_response(self, value):
        """
        Apply the second layer of RAPPOR, Instantaneous Randomized Response (IRR).
        """
        irr = np.random.choice([0, 1], p=[1 - self.prob_q if value == 1 else 1 - self.prob_p, self.prob_q if value == 1 else self.prob_p])
        return irr

    def collect_data(self, client_data):
        """
        Simulate data collection from clients.
        """
        aggregated_data = np.zeros((self.STATE_COUNT))
        for i, value in enumerate(client_data):
            prr = self._apply_permanent_randomized_response(value)
            for j in range(self.collection_count):
                irr = self._apply_instantaneous_randomized_response(prr)
                aggregated_data[i] += irr
        return aggregated_data
    
    def get_eps(self):
        raise RuntimeError("Privacy budget us not available for RAPPOR")

    def gen_random_output(self, actual_value, eps, prob_f, is_eps = False):
        try:
            index_of_actual_value = self.INPUT_ALPHABET.index(actual_value)
        except ValueError:
            print(f"{actual_value} is not in the alphabet")

        if is_eps:
            k = np.exp(eps/2)
            self.prob_f = 2/(1+k)
        else:
            self.prob_f = prob_f
            
        bloom_filter = np.zeros(self.STATE_COUNT, dtype=int)
        bloom_filter[index_of_actual_value] = 1
        pertubed_value = self.collect_data(bloom_filter)
        # print("actual ", bloom_filter, " pertubed_value ", pertubed_value)
        perturbed_ = [self.INPUT_ALPHABET[np.argmax(pertubed_value)]]
        # print("Actual ", actual_value, "perturbed_ ", perturbed_)
        return perturbed_
        

    def get_mechanism(self, eps):
        raise RuntimeError("RAPPOR does not support this method.")

    def get_name(self):
        return "Rappor Mechanism"
