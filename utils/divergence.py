from utils.util_functions import *

class Divergence:
    def __init__(self, type):
        self.DIV_TYPE = type
        
    def cal_divergence(self, p, q):
        validate_distribution(prior_dist=p, n=len(q))
        validate_distribution(prior_dist=q, n=len(p))
        
        if self.DIV_TYPE == "TV":
            return cal_tv(p, q)
        else:
            raise RuntimeError(f"Not implemented {self.DIV_TYPE}")
