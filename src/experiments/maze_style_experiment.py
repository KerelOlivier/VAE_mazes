from src.MazeDataset import MazeDataset

from collections import namedtuple
import numpy as np

import torch
from torch import nn
import pandas as pd

class StyleExperiment:
    def __init__(self, 
                 models:list[nn.Module] | list[tuple[nn.Module]], 
                 datasets:list[MazeDataset],
                 metrics:list[callable],
                 n:int=100,
                 device=None,
                 path="results/maze_style_experiment.csv",
                 ) -> None:
        """
        Given m >=1 different models M = {M_0,...M_m-1}, 
        each trained on a different dataset D_i in D = {D_0,...,D_m-1}.
        - Also supports tuples of models TM = (M_0, ... M_i) instead of M_0
            Use tuples if you have multiple models trained on one dataset, and want to compare

        Compute the following metrics for n>=1 samples generated by each model:
        - Mean branching factor
        - Number of connected componentes
        - Node degree histogram
        - Has path s -> e
        - Has 1 entrance and 1 exit in the outer wall
        - Keeps path y in final sample
        - TODO: AVERAGE SHORTEST PATH LENGTH
        - TODO: NUMBER OF CYCLES
        - TODO: CURL TO STRAIGHT RATIO

        Compute the same statistics for n>=1 reconstructions

        Compute the same statistics for n>=1 samples from the dataset


        Reports:
        - Csv file of sample stats
            - how the generated sample stats compare to the real dataset statistics
        - Csv file of reconstruction stats
            - how the reconstruction stats compare to the real dataset statistics

        Args:
            models: list[nn.Module] | list[tuple[nn.Module]]; List of models M = {M_0,...M_m-1}
            datasets: list[MazeDataset]; List of datasets D = {D_0,...,D_m-1}
            metrics: list[callable]; List of metrics to compute
            n: int; Number of samples/reconstructions to generate
            device: torch.device; Device to use for computation
        """
        self.models = models
        self.datasets = datasets
        self.metrics = metrics
        self.n = n
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.path = path
        # make a named tuple 'Row' with keys for each metric
        self.row_template = namedtuple('Row', ['Name'] + [metric.__name__ for metric in self.metrics])

    def run(self):
        """
        Run the experiment. Produce csv files of sample and reconstruction statistics.
        """
        sample_rows = []
        reconstruction_rows = []
        for i, dataset in enumerate(self.datasets):
            # compute ground truth statistics
            gt_row = self.compute_stats(dataset, None, self.metrics, name=f"Ground truth {dataset.name}")
            sample_rows.append(gt_row)
            reconstruction_rows.append(gt_row)
            # if self.models[i] is a tuple -> another for loop
                # for each model in the tuple (Mij), 
                # compute statistics for Mij samples/reconstructions 
            if isinstance(self.models[i], tuple):
                # For each model in the tuple, compute statistics for samples and reconstructions
                for model in self.models[i]:
                    samples, sample_paths = self.make_n_samples(model=model, dataset=dataset, n=self.n)
                    reconstructions, reconstruction_paths = self.make_n_reconstructions(model=model, dataset=dataset, n=self.n)
                    sample_rows.append(self.compute_stats(samples, sample_paths, self.metrics, name=f"{model.name} samples"))
                    reconstruction_rows.append(self.compute_stats(reconstructions, reconstruction_paths, self.metrics, name=f"{model.name} reconstructions"))
            # otherwise, compute statistics for Mi samples/reconstructions
            else:
                # For each model, compute statistics for samples and reconstructions
                model = self.models[i]
                samples, sample_paths = self.make_n_samples(model=model, dataset=dataset, n=self.n)
                reconstructions, reconstruction_paths = self.make_n_reconstructions(model=model, dataset=dataset, n=self.n)
                sample_rows.append(self.compute_stats(samples, sample_paths, self.metrics, name=f"{model.name} samples"))
                reconstruction_rows.append(self.compute_stats(reconstructions, reconstruction_paths, self.metrics, name=f"{model.name} reconstructions"))

        # save sample and reconstruction statistics to csv
        sample_df = pd.DataFrame(sample_rows)
        reconstruction_df = pd.DataFrame(reconstruction_rows)
        # append the results to the csv file
        sample_df.to_csv(self.path, mode='a', header=True)
        reconstruction_df.to_csv(self.path, mode='a', header=True)

    def make_n_samples(self, model:nn.Module, dataset:MazeDataset, n:int=100):
        """
        Make n samples from the VAE.

        Args:
            model: nn.Module; VAE model
            dataset: MazeDataset; Dataset to sample from
            n: int; Number of samples to generate
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

    def make_n_reconstructions(self, model:nn.Module, dataset:MazeDataset, n:int=100):
        """
        Reconstruct n samples from the VAE.

        Args:
            model: nn.Module; VAE model
            dataset: MazeDataset; Dataset to sample from
            n: int; Number of samples to generate
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
        return reconstructions, y

    def compute_stats(self, mazes:np.ndarray | MazeDataset, paths:np.ndarray | None, metrics, name:str=None, **kwargs):
        """
        Compute the metrics for the given mazes.

        Args:
            mazes: np.ndarray | MazeDataset; Mazes to compute metrics for
            metrics: list[callable]; Metrics to compute
            name: str; Name of the dataset (optional)
            kwargs: dict; Additional arguments to pass to the metrics
        """
        if isinstance(mazes, MazeDataset):
            # Select n random samples from the dataset
            random_idx = np.random.choice(np.arange(len(mazes)), size=self.n)
            X = []
            Y = []
            for idx in random_idx:
                x, y = mazes[idx]
                # Convert to square maze
                if len(x.shape) == 1:
                    x = x.view(int(np.sqrt(x.shape[0])), int(np.sqrt(x.shape[0])))
                if len(y.shape) == 1:
                    y = y.view(int(np.sqrt(y.shape[0])), int(np.sqrt(y.shape[0])))
                X.append(x)
                Y.append(y)
                
            mazes = np.stack(X)
            mazes = mazes.astype(np.int32)
            mazes = mazes.squeeze(1)
            paths = np.stack(Y)
            paths = paths.astype(np.int32)
            paths = paths.squeeze(1)
        else:
            mazes = mazes.detach().cpu().numpy()
            if len(mazes.shape) == 2:
                width, height = int(np.sqrt(mazes.shape[1])), int(np.sqrt(mazes.shape[1]))
                mazes = mazes.reshape(-1, width, height)
                mazes = mazes.astype(np.int32)
                if len(mazes.shape) == 4:
                    mazes = mazes.squeeze(1)

            if paths is not None:
                paths = paths.detach().cpu().numpy()
                paths = paths.reshape(-1, width, height)
                paths = paths.astype(np.int32)
                if len(paths.shape) == 4:
                    paths = paths.squeeze(1)

        # Compute the metrics for the mazes
        row_list = [name]
        for metric in metrics:
            row_list.append(metric(mazes=mazes, paths=paths))
        
        row = self.row_template(*row_list)
        return row
    