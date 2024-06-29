from src.MazeDataset import MazeDataset

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import os

class VisualExperiment:
    def __init__(self, 
                models:list[nn.Module] | list[tuple[nn.Module]], 
                datasets:list[MazeDataset],
                device=None,
                path="figures/",
                n:int=4
                ) -> None:
        """
        Given m >=1 different models M = {M_0,...M_m-1},
        each trained on a different dataset D_i in D = {D_0,...,D_m-1}.

        Plot samples and reconstructions of each model on each dataset.

        Args:
            models: list[nn.Module] | list[tuple[nn.Module]]; List of models M = {M_0,...M_m-1}
            datasets: list[MazeDataset]; List of datasets D = {D_0,...,D_m-1}
            device: torch.device; Device to use for computation
            path: str; Path to save the results to
        """
        self.models = models
        self.datasets = datasets
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.path = path
        self.n = n
    
    def run(self):
        dataset_dir_lookup = {
            "RandomizedDFS":"dfs",
            "Prim":"prim",
            "FractalTessellation":"fractal"
        }
        # If all datasets are included, plot a cover page
        if len(self.datasets) == 3:
            self.plot_cover_page(self.datasets)
        # Iterate over the datasets
        for i, dataset in enumerate(self.datasets):
            # For each dataset, iterate over the models
            models = self.models[i]
            # For each (tuple of) model(s), generate samples and reconstructions
            if isinstance(models, tuple):
                for model in models:
                    print(f"VIS: {dataset.name}, {model.name}")
                    # Make n samples and reconstructions
                    samples, _ = self.make_n_samples(model=model, dataset=dataset, n=self.n)
                    reconstructions, y, x = self.make_n_reconstructions(model=model, dataset=dataset, n=self.n)
                    # If the samples are 2D (flattened), reshape them to 3D
                    if len(samples.shape) == 2:
                        width = height = int(np.sqrt(samples.shape[-1]))
                        samples = samples.reshape(-1, width, height)
                        reconstructions = reconstructions.reshape(-1, width, height)
                    # Convert the samples and reconstructions to numpy arrays
                    samples = samples.cpu().detach().numpy()
                    reconstructions = reconstructions.cpu().detach().numpy()

                    stripped_dataset_name = dataset.name.split(" ")[1]
                    # Plot the samples, reconstructions and input mazes
                    self.plot_mazes(samples, f"{model.name} samples on {dataset.name}", self.path+f"{dataset_dir_lookup[stripped_dataset_name]}/", f"{model.name}_samples.png")
                    self.plot_mazes(reconstructions, f"{model.name} reconstructions on {dataset.name}", self.path+f"{dataset_dir_lookup[stripped_dataset_name]}/", f"{model.name}_reconstructions.png")
                    self.plot_mazes(x.cpu().detach().numpy(), f"{model.name} input mazes on {dataset.name}", self.path+f"{dataset_dir_lookup[stripped_dataset_name]}/", f"{model.name}_input_mazes.png")
                    
            else:
                model = models[i]
                # Make n samples and reconstructions
                samples, _ = self.make_n_samples(model=model, dataset=dataset, n=self.n)
                reconstructions, y, x = self.make_n_reconstructions(model=model, dataset=dataset, n=self.n)
                # If the samples are 2D (flattened), reshape them to 3D
                if len(samples.shape) == 2:
                    width = height = int(np.sqrt(samples.shape[-1]))
                    samples = samples.reshape(-1, width, height)
                    reconstructions = reconstructions.reshape(-1, width, height)
                # Convert the samples and reconstructions to numpy arrays
                samples = samples.cpu().detach().numpy()
                reconstructions = reconstructions.cpu().detach().numpy()

                stripped_dataset_name = dataset.name.split(" ")[1]
                # Plot the samples, reconstructions and input mazes
                self.plot_mazes(samples, f"{model.name} samples on {dataset.name}", self.path+f"{dataset_dir_lookup[stripped_dataset_name]}/", f"{model.name}_samples.png")
                self.plot_mazes(reconstructions, f"{model.name} reconstructions on {dataset.name}", self.path+f"{dataset_dir_lookup[stripped_dataset_name]}/", f"{model.name}_reconstructions.png")
                self.plot_mazes(x.cpu().detach().numpy(), f"{model.name} input mazes on {dataset.name}", self.path+f"{dataset_dir_lookup[stripped_dataset_name]}/", f"{model.name}_input_mazes.png")
                
    def plot_cover_page(self, datasets:list[MazeDataset], title="Style of maze algorithms") -> None:
        """
        Plot a cover page for the visual experiment

        Args:
            datasets: list[MazeDataset]; The datasets to include in the cover page
        """
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        
        for i, ax in enumerate(axs.flat):
            if i == 3:
                break
            idx = np.random.randint(0, len(datasets))
            maze, path = datasets[i].__getitem__(idx)
            maze = maze.squeeze(0)
            path = path.squeeze(0)
            ax = self.plot_maze(ax, maze, path, datasets[i].name.split(" ")[1])

        maze, path = datasets[0].__getitem__(0)
        maze = maze.squeeze(0)
        path = path.squeeze(0)
        ax = self.plot_maze_and_path(axs[1, 1], maze, path, datasets[0].name.split(" ")[1])
        axs[1, 1] = ax
        
        fig.suptitle(title, fontsize=40)
        plt.savefig(self.path+"cover_page.png")
        plt.close()

    def plot_maze(self, ax, maze:np.ndarray, path:np.ndarray, title:str) -> None:
        # plot 1s as black squares
        # and 0s as white squares
        # use inverse cmap to make 0s white
        ax.imshow(maze, cmap='gray_r')
        # make tuples from neighboring 1's in the path
        # and plot them as a line
        for i in range(1, path.shape[0]-1):
            for j in range(1, path.shape[1]-1):
                if path[i, j] == 1:
                    if path[i-1, j] == 1:
                        ax.plot([j, j], [i-1, i], 'w')
                    if path[i+1, j] == 1:
                        ax.plot([j, j], [i, i+1], 'w')
                    if path[i, j-1] == 1:
                        ax.plot([j-1, j], [i, i], 'w')
                    if path[i, j+1] == 1:
                        ax.plot([j, j+1], [i, i], 'w')

        # for the entrance and exit points
        # also draw the line to the edge of the maze
        # to make it look more complete
        
        # find the entrance and exit points (1s on the edge of the maze)

        for i in range(path.shape[0]):
            if path[i, 0] == 1:
                entrance = (0, i)
                ax.plot([entrance[0]-0.5, entrance[0]], [entrance[1], entrance[1]], 'w')
            if path[i, -1] == 1:
                exit_ = (path.shape[1]-1, i)
                ax.plot([exit_[0], exit_[0]+0.5], [exit_[1], exit_[1]], 'r')

        for j in range(path.shape[1]):
            if path[0, j] == 1:
                entrance = (j, 0)
                ax.plot([entrance[0], entrance[0]], [entrance[1]-0.5, entrance[1]], 'w')
            if path[-1, j] == 1:
                exit_ = (j, path.shape[0]-1)
                ax.plot([exit_[0], exit_[0]], [exit_[1], exit_[1]+0.5], 'w')
        
        # draw the entrance and exit lines
        ax.set_title(title, fontsize=25)
        return ax

    def plot_maze_and_path(self, ax, maze:np.ndarray, path:np.ndarray, title:str) -> None:
        LW = 4 if maze.shape[0] < 100 else 2

        # plot 1s as black squares
        # and 0s as white squares
        # use inverse cmap to make 0s white
        ax.imshow(maze, cmap='gray_r')
        # make tuples from neighboring 1's in the path
        # and plot them as a line
        for i in range(1, path.shape[0]-1):
            for j in range(1, path.shape[1]-1):
                if path[i, j] == 1:
                    if path[i-1, j] == 1:
                        ax.plot([j, j], [i-1, i], 'r', linewidth=LW)
                    if path[i+1, j] == 1:
                        ax.plot([j, j], [i, i+1], 'r', linewidth=LW)
                    if path[i, j-1] == 1:
                        ax.plot([j-1, j], [i, i], 'r', linewidth=LW)
                    if path[i, j+1] == 1:
                        ax.plot([j, j+1], [i, i], 'r', linewidth=LW)

        # for the entrance and exit points
        # also draw the line to the edge of the maze
        # to make it look more complete
        
        # find the entrance and exit points (1s on the edge of the maze)

        for i in range(path.shape[0]):
            if path[i, 0] == 1:
                entrance = (0, i)
                ax.plot([entrance[0]-0.5, entrance[0]], [entrance[1], entrance[1]], 'r', linewidth=LW)
            if path[i, -1] == 1:
                exit_ = (path.shape[1]-1, i)
                ax.plot([exit_[0], exit_[0]+0.5], [exit_[1], exit_[1]], 'r', linewidth=LW)

        for j in range(path.shape[1]):
            if path[0, j] == 1:
                entrance = (j, 0)
                ax.plot([entrance[0], entrance[0]], [entrance[1]-0.5, entrance[1]], 'r', linewidth=LW)
            if path[-1, j] == 1:
                exit_ = (j, path.shape[0]-1)
                ax.plot([exit_[0], exit_[0]], [exit_[1], exit_[1]+0.5], 'r', linewidth=LW)
        
        # draw the entrance and exit lines
        ax.set_title(title, fontsize=25)
        return ax

    def plot_mazes(self, mazes:np.ndarray, title:str, file_dir:str, file_name:str) -> None:
        """
        Plot mazes in a grid

        Args:
            mazes: np.ndarray; The mazes to plot
            title: str; The title of the plot
            file_dir: str; The directory to save the plot to
            file_name: str; The name of the file to save the plot to
        """
        if len(mazes.shape) == 4:
            mazes = mazes.squeeze(1)
        
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        rows = cols = int(np.sqrt(mazes.shape[0]))
        if rows == cols == 1:
            fig, ax = plt.subplots(1, 1, figsize=(20, 20))
            ax.imshow(mazes[0], cmap='gray_r')
            ax.set_title(title, fontsize=40)
            plt.savefig(file_dir+file_name)
            plt.close()
            return
        fig, axs = plt.subplots(rows, cols, figsize=(20, 20))
        for i, ax in enumerate(axs.flat):
            ax.imshow(mazes[i], cmap='gray_r')
        # Set suptitle size to 20
        fig.suptitle(title, fontsize=20)
        plt.savefig(file_dir+file_name)
        plt.close()

    def make_n_samples(self, model:nn.Module, dataset:MazeDataset, n:int=16) -> np.ndarray:
        """
        Generate n samples from the model

        Args:
            model: nn.Module; The model to generate samples from
            dataset: MazeDataset; The dataset to take y from
            n: int; The number of samples to generate

        Returns:
            np.ndarray; The samples
        """
        # If the model is not conditional, set y to None
        if not model.is_conditional:
            return model.sample(y=None, batch_size=n), None
        # Select n random samples from the dataset
        random_idx = np.random.choice(np.arange(len(dataset)), size=n)
        Y = []
        for idx in random_idx:
            _, y = dataset[idx]
            Y.append(y)
        y = torch.stack(Y)
        y = y.to(self.device)
        # Generate n samples, given y (optional)
        samples = model.sample(y=y, batch_size=n)
        return samples, y

    def count_node_degrees(self, maze:np.ndarray) -> np.ndarray:
        """
        Count the node degrees in a maze

        Args:
            maze: np.ndarray; The maze to count the node degrees in

        Returns:
            np.ndarray; The node degrees
        """
        node_degrees_count = np.zeros(5)
        for i in range(1, maze.shape[0]-1):
            for j in range(1, maze.shape[1]-1):
                if maze[i, j] == 1:
                    continue
                neighbors = 4 - maze[i-1, j] - maze[i+1, j] - maze[i, j-1] - maze[i, j+1]
                node_degrees_count[neighbors] += 1
        return node_degrees_count

    def aggr_node_distribution(self, dataset:MazeDataset, n=100):
        if isinstance(dataset, MazeDataset):
            mazes = dataset[:, 0]
            # pick n random mazes
            mazes = mazes[:n]
        else:
            mazes = dataset
        
        node_degrees = np.zeros(5)
        for i, maze in enumerate(mazes):
            node_degrees += self.count_node_degrees(maze)
        return node_degrees
    
    def plot_node_distribution(self, ground_truth, model_samples_tuple, title:str, file_dir:str, file_name:str) -> None:
        """
        Plot the node distribution

        Args:
            ground_truth: np.ndarray; The ground truth node distribution
            model_samples: np.ndarray; The node distribution of the model samples
            title: str; The title of the plot
            file_dir: str; The directory to save the plot to
            file_name: str; The name of the file to save the plot to
        """
        # plt.barh(x_axis, list(x_gt), 0.4, label="Ground truth", color="#83cbeb")
        # plt.barh(x_axis+.4, list(x_fc), 0.4, label="Fc VAE", color="#b4e4a2")
        # plt.barh(x_axis+.8, list(x_cn), 0.4, label="Conv VAE", color="#ff7c80")
        # plt.barh(x_axis+1.2, list(x_at), 0.4, label="Transformer VAE", color="#7a4983")
        colors = ["#83cbeb", "#b4e4a2", "#ff7c80", "#7a4983"]
        models = ["Fc VAE", "Conv VAE", "Transformer VAE"]
        x_axis = np.arange(5)
        plt.bar(x_axis, ground_truth, 0.2, label="Ground truth", color="#83cbeb")
        for i, model_samples in enumerate(model_samples_tuple):
            plt.bar(x_axis+.2*(i+1), model_samples, 0.2, label=models[i], color=colors[i+1])
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(file_dir+file_name)
        plt.close()

    def make_n_reconstructions(self, model:nn.Module, dataset:MazeDataset, n:int=16, return_ground_truth=True) -> np.ndarray:
        """
        Generate n reconstructions of the dataset

        Args:
            model: nn.Module; The model to generate reconstructions with
            dataset: MazeDataset; The dataset to reconstruct
            n: int; The number of reconstructions to generate
            return_ground_truth: bool; Whether to return the ground truth

        Returns:
            reconstructions: np.ndarray; The reconstructions
            y: np.ndarray; The paths
            x: np.ndarray; The input mazes (optional)
        """
        # Select n random samples from the dataset
        random_idx = np.random.choice(np.arange(len(dataset)), size=n)
        X = []
        Y = []
        for idx in random_idx:
            x, y = dataset[idx]
            X.append(x)
            Y.append(y)
        x = torch.stack(X)
        y = torch.stack(Y)
        x = x.to(self.device)
        y = y.to(self.device)
        # If the model is not conditional, set y to None
        if not model.is_conditional:
            y = None
        # Reconstruct the samples
        reconstructions = model.auto_encode(x=x, y=y)

        if return_ground_truth:
            return reconstructions, y, x
        return reconstructions, y