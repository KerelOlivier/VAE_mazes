"""
Aggregated metrics for ground-truth datasets of mazes or VAE-generated mazes
"""

import numpy as np
from src.experiments.metrics import *

def aggr_node_degrees_histogram(mazes, aggr='mean', **kwargs):
    """
    Compute the branching factor of a dataset of mazes

    Args:
        mazes (np.ndarray): The dataset of mazes to compute the branching factor of
        aggr (str): The aggregation function to use ('sum' or 'mean')
    
    Returns:
        float: The branching factor of the dataset
    """
    node_degrees = np.zeros(5)
    for _, maze in enumerate(mazes):
        node_degrees += node_degrees_histogram(maze)

    if aggr == 'sum':
        return node_degrees
    elif aggr == 'mean':
        return node_degrees / sum(node_degrees)

def aggr_branching_factor(mazes, aggr='mean', **kwargs):
    """
    Compute the branching factor of a dataset of mazes

    Args:
        mazes (np.ndarray): The dataset of mazes to compute the branching factor of
        aggr (str): The aggregation function to use ('sum' or 'mean')
    
    Returns:
        float: The branching factor of the dataset
    """
    return aggr_node_degrees_histogram(mazes, aggr=aggr, **kwargs)

def aggr_connected_components(mazes, aggr='mean', **kwargs):
    """
    Count the number of connected components in a dataset of mazes

    Args:
        mazes (np.ndarray): The dataset of mazes to count the connected components of
        aggr (str): The aggregation function to use ('sum' or 'mean')
    
    Returns:
        float: The number of connected components in the dataset
    """
    connected_components = 0
    for _, maze in enumerate(mazes):
        connected_components += connected_components(maze)

    if aggr == 'sum':
        return connected_components
    elif aggr == 'mean':
        return connected_components / mazes.shape[0]

def aggr_count_holes_in_outer_wall(mazes, aggr='mean', **kwargs):
    """
    Count the number of holes in the outer wall of a dataset of mazes

    Args:
        mazes (np.ndarray): The dataset of mazes to count the holes of
        aggr (str): The aggregation function to use ('sum' or 'mean')
    
    Returns:
        float: The aggregated number of holes in the dataset
    """
    holes = 0
    for _, maze in enumerate(mazes):
        holes += count_holes_in_outer_wall(maze)

    if aggr == 'sum':
        return holes
    elif aggr == 'mean':
        return holes / mazes.shape[0]
    
def aggr_cycles(mazes, aggr='mean', **kwargs):
    """
    Count the number of cycles in a dataset of mazes

    Args:
        maze (np.ndarray): The dataset of mazes to count the cycles of
        aggr (str): The aggregation function to use ('sum' or 'mean')
    
    Returns:
        float: The aggregated number of cycles in the dataset
    """
    cycles = 0
    for _, maze in enumerate(mazes):
        cycles += cycles(maze)

    if aggr == 'sum':
        return cycles
    elif aggr == 'mean':
        return cycles / mazes.shape[0]
    
def aggr_has_path(mazes, paths, aggr='mean', **kwargs):
    """
    Check in how many mazes of a dataset there is a path from the start to the goal

    Args:
        mazes (np.ndarray): The mazes to check for a path
        paths (np.ndarray): The paths to check for a path
        aggr (str): The aggregation function to use ('sum' or 'mean')
    
    Returns:
        float: The number of mazes with a path from the start to the goal
    """
    n_paths = 0
    for i, maze in enumerate(mazes):
        # start id and target id are always on the outer wall
        outer_wall_mask = np.zeros(maze.shape)
        outer_wall_mask[0, :] = 1
        outer_wall_mask[-1, :] = 1
        outer_wall_mask[:, 0] = 1
        outer_wall_mask[:, -1] = 1

        # find start and target id by looking where paths[i] == 1 and outer_wall_mask == 1
        exits = np.where(paths[i] * outer_wall_mask == 1)
        if len(exits) == 0:
            continue
        start_idx = exits[0][0]
        target_idx = exits[1][0]


        if has_path(maze, start_idx, target_idx):
            n_paths += 1

    if aggr == 'sum':
        return n_paths
    elif aggr == 'mean':
        return n_paths / mazes.shape[0]
    
def aggr_keeps_shortest_path(mazes, paths, aggr='sum', **kwargs):
    """
    Check in how many mazes of a dataset the shortest path (y) is kept after generation

    Args:
        mazes (np.ndarray): The mazes to check for the shortest path
        paths (np.ndarray): The paths in question
        aggr (str): The aggregation function to use ('sum' or 'mean')

    Returns:
        float: The aggregated number of mazes where the shortest path is kept
    """
    n_kept = 0
    for i, maze in enumerate(mazes):
        if keeps_shortest_path(maze, paths[i]):
            n_kept += 1

    if aggr == 'sum':
        return n_kept
    elif aggr == 'mean':
        return n_kept / mazes.shape[0]
    
def aggr_average_shortest_path_length(mazes, aggr='sum', **kwargs):
    """
    Compute the average shortest path length of a dataset of mazes

    Args:
        mazes (np.ndarray): The dataset of mazes to compute the average shortest path length of
        aggr (str): The aggregation function to use ('sum' or 'mean')
    
    Returns:
        float: The average shortest path length of the dataset
    """
    lengths = 0
    for _, maze in enumerate(mazes):
        lengths += average_shortest_path_length(maze)

    if aggr == 'sum':
        return lengths
    elif aggr == 'mean':
        return lengths / mazes.shape[0]
