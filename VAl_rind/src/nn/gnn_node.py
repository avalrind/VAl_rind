import Parameters
import numpy as np

class gnn_node : 
    
    def __init__(self , 
                 name : str , 
                 connections : list , 
                 is_inp_node : bool , 
                 is_out_node : bool) : 
        
        self.name = name
        self.connections = connections
        self.is_inp_node = is_inp_node
        self.is_out_node = is_out_node

        self.value = np.random.rand(1)
        
        self.weight = np.random.rand(1)
        self.bias = np.random.rand(1)

        Parameters('weight' , self.weight)
        Parameters('bias' , self.bias)

    add_connection = lambda self , nodes : self.connections.extend(nodes)