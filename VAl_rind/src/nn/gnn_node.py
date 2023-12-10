import Parameters , Module
import numpy as np

class gnn_node(Module) : 
    '''
    gnn_node class for the GNN model. Inherits from the Module class

    Parameters

        1) name : Name of the node
        2) connections : Connections of the node
        3) is_inp_node : Whether the node is an input node
        4) is_out_node : Whether the node is an output node
    '''
    
    def __init__(self , 
                 name : str , 
                 connections : list , 
                 is_inp_node : bool , 
                 is_out_node : bool) : 
        
        '''
        Constructor for the gnn_node class
        '''
        
        self.name = name
        self.connections = connections
        self.is_inp_node = is_inp_node
        self.is_out_node = is_out_node

        self.value = np.random.rand(1)
        
        self.weight = np.random.rand(1)
        self.bias = np.random.rand(1)

        Parameters('weight' , self.weight)
        Parameters('bias' , self.bias)

    def add_connection(self , nodes : list) : 
        '''
        Adds a connection to the node

        Parameters

            1) nodes : Nodes to be added
        '''

        self.connections.extend(nodes)
