import torch
from torch_geometric.data import Data

import torch

import numpy as np

#Temporal metrics for dataset statistics.

def compute_edge_recurrency(test_set, train_set):
    def concatenate_edge_indices(snapshots):
        # Initialize an empty list to store the concatenated edge indices
        concatenated_edge_index = []
        
        for snap in snapshots:
            # Append the edge_index to the list
            concatenated_edge_index.append(snap.edge_index)
        
        # Concatenate all edge indices along the 1st axis (edges axis)
        concatenated_edge_index = torch.cat(concatenated_edge_index, dim=1)
        
        return concatenated_edge_index

    """
    Compute the portion of edges in test_set_index that are already seen in train_index.
    
    Parameters:
    - test_set_index: A tensor of shape [2, num_edges] representing edges in the test set.
    - train_index: A tensor of shape [2, num_edges] representing edges in the train set.
    
    Returns:
    - recurrent_count: The number of edges in test_set_index that are present in train_index.
    - total_test_edges: The total number of edges in test_set_index.
    - recurrence_ratio: The ratio of recurrent edges to total test edges.
    """
    a = concatenate_edge_indices(test_set)
    b = concatenate_edge_indices(train_set)
    
    # Convert edge indices to sets of tuples for comparison
    a_set = set(map(tuple, a.T.tolist()))  # Convert test edges to a set
    b_set = set(map(tuple, b.T.tolist()))  # Convert train edges to a set

    # Compute the intersection of edges
    common_elements = a_set & b_set  # Common edges between test and train sets
    
    # Calculate the ratio
    recurrent_count = len(common_elements)
    total_test_edges = len(a_set)
    recurrence_ratio = recurrent_count / total_test_edges if total_test_edges > 0 else 0
    
    return recurrence_ratio


def compute_edge_reciprocity(test_set, train_set):
    def concatenate_edge_indices(snapshots):
        # Initialize an empty list to store the concatenated edge indices
        concatenated_edge_index = []
        
        for snap in snapshots:
            # Append the edge_index to the list
            concatenated_edge_index.append(snap.edge_index)
        
        # Concatenate all edge indices along the 1st axis (edges axis)
        concatenated_edge_index = torch.cat(concatenated_edge_index, dim=1)
        
        return concatenated_edge_index

    """
    Compute the portion of edges in test_set_index that are already seen in train_index.
    
    Parameters:
    - test_set_index: A tensor of shape [2, num_edges] representing edges in the test set.
    - train_index: A tensor of shape [2, num_edges] representing edges in the train set.
    
    Returns:
    - recurrent_count: The number of edges in test_set_index that are present in train_index.
    - total_test_edges: The total number of edges in test_set_index.
    - recurrence_ratio: The ratio of recurrent edges to total test edges.
    """
    a = concatenate_edge_indices(test_set)
    b = concatenate_edge_indices(train_set)

    a = a.flip(0) #swap the edge in test set and count their recurrence
    
    # Convert edge indices to sets of tuples for comparison
    a_set = set(map(tuple, a.T.tolist()))  # Convert test edges to a set
    b_set = set(map(tuple, b.T.tolist()))  # Convert train edges to a set

    # Compute the intersection of edges
    common_elements = a_set & b_set  # Common edges between test and train sets
    
    # Calculate the ratio
    recurrent_count = len(common_elements)
    total_test_edges = len(a_set)
    recurrence_ratio = recurrent_count / total_test_edges if total_test_edges > 0 else 0
    
    return recurrence_ratio
    
def compute_structural_homophily(train_set):

    def concatenate_edge_indices(snapshots):
        # Initialize an empty list to store the concatenated edge indices
        concatenated_edge_index = []
        
        for snap in snapshots:
            # Append the edge_index to the list
            concatenated_edge_index.append(snap.edge_index)
        
        # Concatenate all edge indices along the 1st axis (edges axis)
        concatenated_edge_index = torch.cat(concatenated_edge_index, dim=1)
        
        return concatenated_edge_index

    def compute_jaccard(edges, edge_index):
        """
        Computes the average Jaccard similarity for all edges in the edge_index.
    
        Args:
            edge_index (torch.Tensor): 2D tensor of shape [2, num_edges], where each column represents an edge.
    
        Returns:
            float: Average Jaccard similarity of the edges.
        """
        # Create adjacency list from edge_index
        num_nodes = max(torch.max(edge_index[0]).item(), torch.max(edge_index[1]).item()) + 1
        neighbors = {i: set() for i in range(num_nodes)}
        for u, v in edge_index.t():
            neighbors[u.item()].add(v.item())
            neighbors[v.item()].add(u.item())
        
        # Compute Jaccard similarity for each edge
        jaccard_similarities = []
        for u, v in edges.t():
            u, v = u.item(), v.item()
            intersection = neighbors[u].intersection(neighbors[v])
            #union = neighbors[u].union(neighbors[v])
            denom = min(len(neighbors[u]),len(neighbors[v]))-1
            #jaccard_sim = len(intersection) / union if len(union) > 0 else 0.0
            jaccard_sim = len(intersection) / denom if denom > 0 else 0.0
            #jaccard_sim = len(intersection)
            jaccard_similarities.append(jaccard_sim)
        
        return jaccard_similarities
        
    graph = concatenate_edge_indices(train_set)
    return np.mean(compute_jaccard(train_set[-1].edge_index, graph))