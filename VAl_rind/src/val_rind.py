import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import warnings

layer_norm = lambda value : (value - value.min()) / ((value.max() - value.min()))

def matmul(f_row , f_col , s_row , s_col , funcs = None):
    
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

    value = np.empty(shape = (len(row) , len(col)))

    for row_index in range(len(row)):

        for col_index in range(len(col)):

            value[row_index][col_index] = row[row_index] * col[col_index]

    return value

padding = lambda val , padding_length : np.concatenate([val , 
                                                        np.zeros(shape = padding_length - len(val) , 
                                                                 dtype = np.int8)])

sigmoid = lambda val : (1) / (1 + np.exp(-val))

distance = lambda node_1 , node_2 : abs(
    (node_1.weight + node_1.bias) - (node_2.weight + node_2.bias)
)

def visualize_graph(nodes : list) : 

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
