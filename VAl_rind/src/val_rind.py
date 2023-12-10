import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import warnings

def layer_norm(value) :
    '''
        Applies Layer Normalization to the given value

        Parameters

            1) value : The value to be normalized
    '''

    return (value - value.min()) / (value.max() - value.min()) + 1e-10


def matmul(f_row , f_col , s_row , s_col , funcs = None):
    '''
        Multiplies two matrices and returns the resultant matrix

        Parameters

            1) f_row : The row of the first matrix
            2) f_col : The column of the first matrix
            3) s_row : The row of the second matrix
            4) s_col : The column of the second matrix
            5) funcs : The functions to be used for multiplication
    '''
    
    if len(f_col) != len(s_row) : raise ValueError(f'Cannot multiply array with dimensions {len(f_row)}x{len(f_col)} , {len(s_row)}x{len(s_col)}')
    if funcs :
        
        if len(funcs) < len(f_col) : 
            
            warnings.warn(f'{len(funcs)} < {len(f_col)} : Defining the leftover to be Multiplications')
            
            funcs += [lambda x , y : x * y] * (len(s_col) - len(funcs))
        
        elif len(funcs) > len(f_col) : raise ValueError(f'{len(funcs)} > {len(f_col)} : Length of funcs should be <= Length of Matrix')
    
    else : funcs = [lambda x , y : x * y] * len(s_col)
            
    val = sum(list(map(
        lambda func , x , y : func(x , y) , funcs , f_col , s_row
    )))
    
    col = s_col * val / 2
    row = f_row * 2
    
    return row , col

def matrix_maker(row , col):
    '''
        Creates a matrix of given dimensions

        Parameters

            1) row : The row of the matrix
            2) col : The column of the matrix
    '''

    value = np.empty(shape = (len(row) , len(col)))

    for row_index in range(len(row)):

        for col_index in range(len(col)):

            value[row_index][col_index] = row[row_index] * col[col_index]

    return value

def padding(value , padding_length):
    '''
        Pads the given value with zeros

        Parameters

            1) value : The value to be padded
            2) padding_length : The length of padding
    '''

    return np.concatenate([value , np.zeros(shape = padding_length - len(value) , dtype = np.int8)])

def relu(value):
    '''
        Applies ReLU activation function to the given value

        Parameters

            1) value : The value to be activated
    '''

    return np.maximum(0 , value)

def softmax(value):
    '''
        Applies Softmax activation function to the given value

        Parameters

            1) value : The value to be activated
    '''

    return np.exp(value) / np.sum(np.exp(value))

def tanh(value):
    '''
        Applies Tanh activation function to the given value

        Parameters

            1) value : The value to be activated
    '''

    return np.tanh(value)

def sigmoid(value):
    '''
        Applies Sigmoid activation function to the given value

        Parameters

            1) value : The value to be activated
    '''

    return (1) / (1 + np.exp(-value))

def distance(node_1 , node_2):
    '''
        Calculates the distance between two nodes

        Parameters

            1) node_1 : The first node
            2) node_2 : The second node
    '''

    return abs((node_1.weight + node_1.bias) - (node_2.weight + node_2.bias))

def visualize_graph(nodes : list) : 
    '''
        Visualizes the given graph

        Parameters

            1) nodes : The nodes of the graph
    '''

    g = nx.Graph()

    for node in nodes : 

        g.add_node(node.name)

        for connection in node.connections : 

            if connection in nodes : g.add_edge(node.name , 
                                                connection.name , 
                                                weight = distance)
                
    pos = nx.spring_layout(g)
    nx.draw(g, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_color='black', font_size=10, edge_color='gray', width=1.0)
    edge_labels = nx.get_edge_attributes(g, 'weight')
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_color='red')

    plt.show()
