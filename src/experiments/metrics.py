import numpy as np
from collections import deque


def mean_branching_factor(maze:np.ndarray):
    """
    Returns the mean branching factor of a maze

    Args:
        maze (np.ndarray): The maze to find the branching factor of
    
    Returns:
        float: The mean branching factor of the maze
    """
    return mean_node_degree(maze)

def connected_components(maze:np.ndarray):
    """
    Use Depth First Search to find the number of connected components in a maze

    Args:
        maze (np.ndarray): The maze to find the connected components in
    
    Returns:
        int: The number of connected components in the maze
    """
    # Make a set of unvisited indices
    unvisited_nodes = set()
    for i, j in np.argwhere(maze == 0):
        unvisited_nodes.add((i,j))
    # Initialize the number of connected components
    n_connected_components = 0

    visited_nodes = set()

    while len(unvisited_nodes) > 0:
        # Increment the number of connected components
        n_connected_components += 1
        # Find a random node
        for random_node in unvisited_nodes:
            break
        
        stack = deque([random_node])
        # DFS until no more cells can be reached from current connected component
        while stack:
            node_idx = stack.pop()
            if node_idx in visited_nodes:
                continue
            unvisited_nodes.remove(node_idx)
            visited_nodes.add(node_idx)
            # check for neighbors
            i, j = node_idx
            if i > 0 and (i-1, j) in unvisited_nodes:
                stack.append((i-1, j))
            if i < maze.shape[0]-1 and (i+1, j) in unvisited_nodes:
                stack.append((i+1, j))
            if j > 0 and maze[i][j-1] == 0 and (i, j-1) in unvisited_nodes:
                stack.append((i, j-1))
            if j < maze.shape[1]-1 and (i, j+1) in unvisited_nodes:
                stack.append((i, j+1))

    return n_connected_components

def mean_node_degree(maze:np.ndarray):
    """
    Compute the mean node degree of a maze,
    which is the average number of neighbors per node.

    Args:
        maze (np.ndarray): The maze to find the mean node degree of
    
    Returns:
        float: The mean node degree of the maze
    """
    mean_node_degree = 0
    for i in range(1, maze.shape[0]-1):
        for j in range(1, maze.shape[1]-1):
            if maze[i, j] == 1:
                
                continue
            neighbors = 4 - maze[i-1, j] - maze[i+1, j] - maze[i, j-1] - maze[i, j+1]
            mean_node_degree += neighbors

    mean_node_degree /= (maze.shape[0]*maze.shape[1] - np.sum(maze))
    return mean_node_degree


def node_degrees_histogram(maze:np.ndarray):
    """
    Compute the node degrees histogram of a maze,
    which is the distribution of the number of neighbors per node.

    Args:
        maze (np.ndarray): The maze to find the node degrees histogram of

    Returns:
        np.ndarray: The node degrees histogram of the maze
    """
    node_degrees_count = np.zeros(5)
    for i in range(1, maze.shape[0]-1):
        for j in range(1, maze.shape[1]-1):
            if maze[i, j] == 1:
                continue
            neighbors = 4 - maze[i-1, j] - maze[i+1, j] - maze[i, j-1] - maze[i, j+1]
            node_degrees_count[neighbors] += 1
    return node_degrees_count

def has_path(maze:np.ndarray, start_idx:tuple[int,int], target_idx:tuple[int,int]):
    """
    DFS to find a path from start to target in a maze
    """
    stack = deque([start_idx])
    visited = set()

    while stack:
        node_idx = stack.pop()
        if node_idx == target_idx:
            return True
        visited.add(node_idx)
        i, j = node_idx
        if i > 0 and maze[i-1, j] == 0 and (i-1, j) not in visited:
            stack.append((i-1, j))
        if i < maze.shape[0]-1 and maze[i+1, j] == 0 and (i+1, j) not in visited:
            stack.append((i+1, j))
        if j > 0 and maze[i, j-1] == 0 and (i, j-1) not in visited:
            stack.append((i, j-1))
        if j < maze.shape[1]-1 and maze[i, j+1] == 0 and (i, j+1) not in visited:
            stack.append((i, j+1))

    return False

def count_holes_in_outer_wall(maze:np.ndarray):
    """
    Count the number of holes in the outer wall of a maze

    Args:
        maze (np.ndarray): The maze to count the holes in the outer wall of
    
    Returns:
        int: The number of holes in the outer wall of the maze
    """
    holes = 0
    for i in range(maze.shape[0]):
        if maze[i, 0] == 0:
            holes += 1
        if maze[i, maze.shape[1]-1] == 0:
            holes += 1
    for j in range(maze.shape[1]):
        if maze[0, j] == 0:
            holes += 1
        if maze[maze.shape[0]-1, j] == 0:
            holes += 1
    return holes

def average_shortest_path_length(maze:np.ndarray):
    """
    Compute the average shortest path length in a maze

    Args:
        maze (np.ndarray): The maze to compute the average shortest path length of
    
    Returns:
        float: The average shortest path length in the maze
    """
    # TODO: @KerelOlivier
    return 0

def cycles(maze:np.ndarray):
    """
    Count the number of cycles in a maze

    Args:
        maze (np.ndarray): The maze to count the cycles in
    
    Returns:
        int: The number of cycles in the maze
    """
    n_cycles = 0
    # TODO: @KerelOlivier
    
    return n_cycles

def keeps_shortest_path(maze:np.ndarray, path:np.ndarray):
    """
    Check if the maze keeps the shortest path

    Args:
        maze (np.ndarray): The maze to check if it keeps the shortest path
        path (np.ndarray): The path in question
    
    Returns:
        bool: True if the maze keeps the shortest path, False otherwise
    """
    path_indices = np.argwhere(path==1)

    for i, j in path_indices:
        if maze[i, j] == 1:
            return False
    return True

def ratio_straight_to_curl_paths(maze:np.ndarray):
    """
    Compute the ratio of straight paths to curl paths in a maze

    Args:
        maze (np.ndarray): The maze to compute the ratio of straight to curl paths of
    
    Returns:
        float: The ratio of straight paths to curl paths in the maze
    """
    straight_paths = 0
    curl_paths = 0
    for i in range(1, maze.shape[0]-1):
        for j in range(1, maze.shape[1]-1):
            if maze[i, j] == 1:
                continue
            straight_path = 0
            curl_path = 0

            if maze[i-1, j] == 0 and maze[i+1, j] == 0:
                straight_path += 1
            if maze[i, j-1] == 0 and maze[i, j+1] == 0:
                straight_path += 1
            if 4 - maze[i-1, j] - maze[i+1, j] - maze[i, j-1] - maze[i, j+1] == 1:
                straight_path += 1
            
            if maze[i-1, j] == 0 and maze[i, j-1] == 0:
                curl_path += 1
            if maze[i-1, j] == 0 and maze[i, j+1] == 0:
                curl_path += 1
            if maze[i+1, j] == 0 and maze[i, j-1] == 0:
                curl_path += 1
            if maze[i+1, j] == 0 and maze[i, j+1] == 0:
                curl_path += 1

            straight_paths += straight_path
            curl_paths += curl_path
    
    return straight_paths / curl_paths
    
if __name__ == "__main__":
    maze = np.array([[0,1,0,0,0],
                     [0,1,0,1,0],
                     [0,0,0,1,0]])
    
    path = np.array([[1,0,1,1,1],
                     [1,0,1,0,1],
                     [1,1,1,0,1]])
    
    # set outside wall around maze to 1
    maze = np.pad(maze, pad_width=1, mode='constant', constant_values=1)
    print(maze)
    path = np.pad(path, pad_width=1, mode='constant', constant_values=0)
    
    print(mean_branching_factor(maze))
    print(connected_components(maze))
    print(mean_node_degree(maze))
    print(node_degrees_histogram(maze))
    print(has_path(maze, (1,1), (3,5)))
    print(count_holes_in_outer_wall(maze))
    print(cycles(maze))
    print(keeps_shortest_path(maze, path))