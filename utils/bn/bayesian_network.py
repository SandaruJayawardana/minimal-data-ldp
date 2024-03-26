import pandas as pd
import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
import networkx as nx
import matplotlib.pyplot as plt

class BN():
    def __init__(self, data, priority_dict = {}):
        self.data = data # Panda database
        self.priority_dict = priority_dict
        self.model = None
        self.attribute_list = list(data.columns)
        self.ordered_attribute_list = None
        self.parent_nodes_dict = None
        priority_dict_keys = priority_dict.keys()

        self.black_list = []
        for i in self.attribute_list:
            for j in self.attribute_list:
                if i == j:
                    continue
                if i not in priority_dict_keys:
                    i_priority = 1
                else:
                    i_priority = priority_dict[i]

                if j not in priority_dict_keys:
                    j_priority = 1
                else:
                    j_priority = priority_dict[j]
                
                if i_priority < j_priority:
                    self.black_list.append((i, j))

    def get_model(self):
        if self.model == None:
            self.train_model()
        return self.model

    def train_model(self):
        hc = HillClimbSearch(self.data)

        # Estimate the best model
        best_model = hc.estimate(scoring_method=BicScore(self.data), max_iter=1000000, black_list=self.black_list)
        print("Best model structure: ", best_model.edges())
        # Define the model with the learned structure
        self.model = BayesianModel(best_model.edges())

        # Learn the parameters using Maximum Likelihood Estimation
        self.model.fit(self.data, estimator=MaximumLikelihoodEstimator)

    def visualize_structure(self):
        def get_level_value(node, nodes_weight_dict, parent_node_dict):
            if len(parent_node_dict[node]) == 0:
                nodes_weight_dict[node] = 0
                return 0
            elif nodes_weight_dict[node] != -1:
                return nodes_weight_dict[node]
            else:
                max_weight = -1
                for i in parent_node_dict[node]:
                    val = get_level_value(i, nodes_weight_dict, parent_node_dict)
                    if max_weight < val:
                        max_weight = val
                nodes_weight_dict[node] = max_weight + 1
                return max_weight + 1

        nodes_weight_dict = {}
        parent_node_dict = {}

        for i in self.attribute_list: #list(model.nodes()):
            nodes_weight_dict[i] = -1
            parent_node_dict[i] = []

        for i in list(self.model.edges()):
            parents = parent_node_dict[i[0]]
            if i[1] not in parents:
                parent_node_dict[i[0]].append(i[1])

        for i in self.attribute_list:
            get_level_value(i, nodes_weight_dict, parent_node_dict)
        self.ordered_attribute_list = sorted(nodes_weight_dict, key=nodes_weight_dict.get) #, reverse=True)

# print("Ordered keys in descending order:", sorted_keys_desc)

        print(len(nodes_weight_dict))
        print(len(parent_node_dict))

        if self.model == None:
            self.train_model()

        G = nx.DiGraph()
        for i in self.attribute_list: #list(model.nodes()):
            G.add_node(i, level=nodes_weight_dict[i]) 

        G.add_edges_from(self.model.edges())
        
        pos = nx.layout.multipartite_layout(G, subset_key="level")
        nx.draw(G, pos, with_labels=True, node_size=3600, node_color="lightblue", font_size=15, font_weight="bold")
        plt.title("Learned Bayesian Network Visualization")
        plt.show()

    def get_ordered_attribute_list(self):
        if self.ordered_attribute_list == None:
            self.visualize_structure()
        return self.ordered_attribute_list
    
    def get_parent_nodes_dict(self):
        if self.parent_nodes_dict == None:
            self.parent_nodes_dict = {}

            for i in self.attribute_list:
                self.parent_nodes_dict[i] = []

            for i in list(self.model.edges()):
                parents = self.parent_nodes_dict[i[1]]
                if i[1] not in parents:
                    self.parent_nodes_dict[i[1]].append(i[0])
        return self.parent_nodes_dict