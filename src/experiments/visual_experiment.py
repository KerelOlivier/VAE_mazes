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
        for dataset in self.datasets:
            for i, models in enumerate(self.models):
                if isinstance(models, tuple):
                    for model in models:
                        samples, _ = self.make_n_samples(model=model, dataset=dataset, n=self.n)
                        reconstructions, y, x = self.make_n_reconstructions(model=model, dataset=dataset, n=self.n)

                        if len(samples.shape) == 2:
                            width = height = int(np.sqrt(samples.shape[-1]))
                            samples = samples.reshape(-1, width, height)
                            reconstructions = reconstructions.reshape(-1, width, height)

                        samples = samples.cpu().detach().numpy()
                        reconstructions = reconstructions.cpu().detach().numpy()

                        stripped_dataset_name = dataset.name.split(" ")[1]
                        self.plot_mazes(samples, f"{model.name} samples on {dataset.name}", self.path+f"{dataset_dir_lookup[stripped_dataset_name]}/", f"{model.name}_samples.png")
                        self.plot_mazes(reconstructions, f"{model.name} reconstructions on {dataset.name}", self.path+f"{dataset_dir_lookup[stripped_dataset_name]}/", f"{model.name}_reconstructions.png")
                        self.plot_mazes(x.cpu().detach().numpy(), f"{model.name} input mazes on {dataset.name}", self.path+f"{dataset_dir_lookup[stripped_dataset_name]}/", f"{model.name}_input_mazes.png")
                
                else:
                    model = models[i]
                    samples, _ = self.make_n_samples(model=model, dataset=dataset, n=self.n)
                    reconstructions, y, x = self.make_n_reconstructions(model=model, dataset=dataset, n=self.n)

                    if len(samples.shape) == 2:
                        width = height = int(np.sqrt(samples.shape[-1]))
                        samples = samples.reshape(-1, width, height)
                        reconstructions = reconstructions.reshape(-1, width, height)

                    samples = samples.cpu().detach().numpy()
                    reconstructions = reconstructions.cpu().detach().numpy()

                    stripped_dataset_name = dataset.name.split(" ")[1]
                    self.plot_mazes(samples, f"{model.name} samples on {dataset.name}", self.path+f"{dataset_dir_lookup[stripped_dataset_name]}/", f"{model.name}_samples.png")
                    self.plot_mazes(reconstructions, f"{model.name} reconstructions on {dataset.name}", self.path+f"{dataset_dir_lookup[stripped_dataset_name]}/", f"{model.name}_reconstructions.png")
                    self.plot_mazes(x.cpu().detach().numpy(), f"{model.name} input mazes on {dataset.name}", self.path+f"{dataset_dir_lookup[stripped_dataset_name]}/", f"{model.name}_input_mazes.png")
    
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
        fig, axs = plt.subplots(rows, cols, figsize=(20, 20))
        for i, ax in enumerate(axs.flat):
            ax.imshow(mazes[i], cmap='gray')
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
        if not model.is_conditional:
            return model.sample(y=None, batch_size=n), None
        
        random_idx = np.random.choice(np.arange(len(dataset)), size=n)
        Y = []
        for idx in random_idx:
            _, y = dataset[idx]
            Y.append(y)
        y = torch.stack(Y)
        y = y.to(self.device)

        samples = model.sample(y=y, batch_size=n)
        return samples, y

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