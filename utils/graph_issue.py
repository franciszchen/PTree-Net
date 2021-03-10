import torch
import numpy as np
import random

"""
[tip
    [mid
        [bottom
            (location/patch/embedding)
        ]
    ]
]
"""

def create_matrix(bottom_list2):
    """
    expand tree topology into graph (A matrix of GCN)
    equation 2
    """
    # [root-0 1-tip_dim mid_dim bottom_dim]
    mid_dim = len(bottom_list2) # 
    bottom_dim = sum([len(bottom_list) for bottom_list in bottom_list2])

    matrix_dim = 1 + mid_dim + bottom_dim
    matrix = torch.zeros(matrix_dim, matrix_dim)
    edge_list = []
    # root
    matrix[0, 0] = 1
    # root and mid
    matrix[0, 1:mid_dim+1] = 1 
    matrix[1:mid_dim+1, 0] = 1 
    for m in range(mid_dim):
        edge_list.append([0, 1+m])
        edge_list.append([1+m, 0])
    # bottom
    previous_bottom_count = 0
    for mid_idx in range(mid_dim):
        # all bottom-patch of a specific mid-patch
        len_bottom_current = len(bottom_list2[mid_idx]) 
        matrix[1+mid_idx, 1+mid_dim+previous_bottom_count:1+mid_dim+previous_bottom_count+len_bottom_current] = 1
        matrix[1+mid_dim+previous_bottom_count:1+mid_dim+previous_bottom_count+len_bottom_current, 1+mid_idx] = 1
        for i in range(len_bottom_current):
            edge_list.append([1+mid_idx, 1+mid_dim+previous_bottom_count+i])
            edge_list.append([1+mid_dim+previous_bottom_count+i, 1+mid_idx])
        # update bottom patch count
        previous_bottom_count += len_bottom_current

    return matrix, edge_list, [mid_dim, bottom_dim]

def pyg_edge_tensor(edge_list):
    edge_tensor = torch.tensor(edge_list, dtype=torch.long)
    return edge_tensor.t().contiguous()

