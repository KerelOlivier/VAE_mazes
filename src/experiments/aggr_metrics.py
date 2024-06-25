"""
Aggregated metrics for ground-truth datasets of mazes or VAE-generated mazes
"""

import numpy as np
from src.experiments.metrics import *

def dataset_node_degrees_histogram(dataset):
    """
    Count the node degrees histogram of a dataset of mazes

    Args:
        dataset (np.ndarray): The dataset of mazes to count the node degrees histogram of
    
    Returns:
        np.ndarray: The node degrees histogram of the dataset
    """
    mazes = dataset[:, 0]
    node_degrees = np.zeros(5)
    for _, maze in enumerate(mazes):
        node_degrees += node_degrees_histogram(maze)
    return node_degrees

