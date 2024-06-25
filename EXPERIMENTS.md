### Experiment 1; Multiple maze generator type
- Multiple model
    - Linear
    - Convolutional
    - Attention

- Multiple types
    * Wilson's (cycle-erased mazes) (default)
    * Recursive backtracker (DFS/BFS) (with horizontal / vertical bias)
    - Prim's
    - Fractal Tessellation

- Metrics
    - connected components
        - also look at number of cells per component (ratio)
    - branching factor
    - distribution of # edges, average shortest path length
    - has path start->end
    - how many have start and end
    - are there cycles?
        - if yes it is adding new properties to maze, unlike spanning tree algs
    - Does it keep the path from start -> end (y)

- Conclusion
    - Which model does the best (on average)
    - Which model does the best for each Maze generator type
    - based on the metrics

### Experiment 2; Uncertainty estimation
- Make a heatmap of uncertainty (distance from closest integer)
    - overlayed on the maze

- see if uncertain areas are worse