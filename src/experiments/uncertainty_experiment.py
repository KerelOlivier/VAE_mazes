from src.MazeDataset import MazeDataset

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import os

class UncertaintyExperiment:
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

        Plot a heatmap of the uncertainty of each model on examples from the respective dataset

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

    def run(self) -> None:
        """Produce 4x4 heatmaps of the uncertainty of each model on examples from the respective dataset"""
        dataset_dir_lookup = {
            "RandomizedDFS":"dfs",
            "Prim":"prim",
            "FractalTessellation":"fractal"
        }
        # Iterate over the datasets
        for i, dataset in enumerate(self.datasets):
            # For each dataset, iterate over the models
            models = self.models[i]
            if isinstance(models, tuple):
                # For each (tuple of) model(s), generate samples and reconstructions
                for model in models:
                    # Make n samples and reconstructions, also compute the uncertainty per pixel
                    samples, _ = self.make_n_samples(model=model, dataset=dataset, n=self.n)
                    sample_uncertainty = self.uncertainty_per_pixel(samples)
                    reconstructions, _ = self.make_n_reconstructions(model=model, dataset=dataset, n=self.n)
                    reconstruction_uncertainty = self.uncertainty_per_pixel(reconstructions)
                    # If the samples are 2D (flattened), reshape them to 3D
                    if len(samples.shape) == 2:
                        width = height = int(np.sqrt(samples.shape[-1]))
                        samples = samples.reshape(-1, width, height)
                        reconstructions = reconstructions.reshape(-1, width, height)
                        sample_uncertainty = sample_uncertainty.reshape(-1, width, height).cpu().detach().numpy()
                        reconstruction_uncertainty = reconstruction_uncertainty.reshape(-1, width, height).cpu().detach().numpy()
                    else:
                        sample_uncertainty = sample_uncertainty.cpu().detach().numpy()
                        reconstruction_uncertainty = reconstruction_uncertainty.cpu().detach().numpy()
                    # Convert the probabilities to samples and reconstructions as numpy arrays
                    samples = torch.bernoulli(samples).cpu().detach().numpy()
                    reconstructions = torch.bernoulli(reconstructions).cpu().detach().numpy()

                    stripped_dataset_name = dataset.name.split(" ")[1]
                    # Plot the samples and reconstructions with the uncertainty as the color
                    self.plot_heatmap(samples, sample_uncertainty, f"{model.name} samples on {dataset.name}", self.path+f"{dataset_dir_lookup[stripped_dataset_name]}/", f"{model.name}_samples_uncertainty.png")
                    self.plot_heatmap(reconstructions, reconstruction_uncertainty, f"{model.name} reconstructions on {dataset.name}", self.path+f"{dataset_dir_lookup[stripped_dataset_name]}/", f"{model.name}_reconstructions_uncertainty.png")
            
            else:
                model = models[i]
                # Make n samples and reconstructions, also compute the uncertainty per pixel
                samples, _ = self.make_n_samples(model=model, dataset=dataset, n=self.n)
                sample_uncertainty = self.uncertainty_per_pixel(samples).cpu().detach().numpy()
                reconstructions, _ = self.make_n_reconstructions(model=model, dataset=dataset, n=self.n)
                reconstruction_uncertainty = self.uncertainty_per_pixel(reconstructions).cpu().detach().numpy()
                # If the samples are 2D (flattened), reshape them to 3D
                if len(samples.shape) == 2:
                    width = height = int(np.sqrt(samples.shape[-1]))
                    samples = samples.reshape(-1, width, height)
                    reconstructions = reconstructions.reshape(-1, width, height)
                    sample_uncertainty = sample_uncertainty.reshape(-1, width, height)
                    reconstruction_uncertainty = reconstruction_uncertainty.reshape(-1, width, height)
                else:
                    sample_uncertainty = sample_uncertainty.cpu().detach().numpy()
                    reconstruction_uncertainty = reconstruction_uncertainty.cpu().detach().numpy()
                # Convert the probabilities to samples and reconstructions as numpy arrays
                samples = torch.bernoulli(samples).cpu().detach().numpy()
                reconstructions = torch.bernoulli(reconstructions).cpu().detach().numpy()

                stripped_dataset_name = dataset.name.split(" ")[1]
                # Plot the samples and reconstructions with the uncertainty as the color
                self.plot_heatmap(samples, sample_uncertainty, f"{model.name} samples on {dataset.name}", self.path+f"{dataset_dir_lookup[stripped_dataset_name]}/", f"{model.name}_samples_uncertainty.png")
                self.plot_heatmap(reconstructions, reconstruction_uncertainty, f"{model.name} reconstructions on {dataset.name}", self.path+f"{dataset_dir_lookup[stripped_dataset_name]}/", f"{model.name}_reconstructions_uncertainty.png")

    def plot_heatmap(self, samples:np.ndarray, uncertainty:np.ndarray, title:str, file_dir:str, file_name:str) -> None:
        """
        Plot a heatmap of the samples with the uncertainty as the color

        Args:
            samples: np.ndarray; The samples to plot
            uncertainty: np.ndarray; The uncertainty of the samples
            title: str; The title of the plot
            file_dir: str; The directory to save the plot to
            file_name: str; The name of the file to save the plot to
        """
        if len(samples.shape) == 4:
            samples = samples.squeeze(1)
        
        if len(uncertainty.shape) == 4:
            uncertainty = uncertainty.squeeze(1)

        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        # Create a grid of the samples
        rows = cells = int(np.sqrt(samples.shape[0]))
        if rows == cells == 1:
            fig, axs = plt.subplots(1, 1, figsize=(20, 20))
            axs.imshow(samples[0], cmap='gray_r')
            axs.imshow(uncertainty[0], cmap='viridis', alpha=0.5)
            fig.colorbar(axs.imshow(uncertainty[0], cmap='viridis', alpha=0.5), ax=axs, orientation='horizontal', fraction=0.05, pad=0.05)
            # fig.suptitle(title, fontsize=40)
            axs.set_yticks([])
            axs.set_xticks([])
            plt.savefig(file_dir+file_name)
            plt.close()
            return
        fig, axs = plt.subplots(rows, cells, figsize=(20, 20))
        for i, ax in enumerate(axs.flat):
            ax.imshow(samples[i], cmap='gray_r')
            ax.imshow(uncertainty[i], cmap='viridis', alpha=0.5)
        
        # show colorbar
        fig.colorbar(axs[0, 0].imshow(uncertainty[0], cmap='viridis', alpha=0.5), ax=axs, orientation='horizontal', fraction=0.05, pad=0.05)
        # Set suptitle size to 20
        fig.suptitle(title, fontsize=20)
        plt.savefig(file_dir+file_name)
        plt.close()

    def uncertainty_per_pixel(self, maze_probabilities:np.ndarray) -> np.ndarray:
        """
        Compute the uncertainty per pixel of the maze probabilities

        Args:
            maze_probabilities: np.ndarray; The probabilities of the maze

        Returns:
            np.ndarray; The uncertainty per pixel
        """
        return 1 - 2 * torch.abs(0.5 - maze_probabilities)

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
        # Set the final sample layer to the identity function (so we get the probabilities)
        model.decoder.final_sample = nn.Identity()
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

    def make_n_reconstructions(self, model:nn.Module, dataset:MazeDataset, n:int=16) -> np.ndarray:
        """
        Generate n reconstructions of the dataset

        Args:
            model: nn.Module; The model to generate reconstructions with
            dataset: MazeDataset; The dataset to reconstruct
            n: int; The number of reconstructions to generate

        Returns:
            np.ndarray; The reconstructions
        """
        # Set the final sample layer to the identity function (so we get the probabilities)
        model.decoder.final_sample = nn.Identity()
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
        return reconstructions, y