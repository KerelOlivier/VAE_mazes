import argparse
import torch

from src.experiments.maze_style_experiment import StyleExperiment
from src.experiments.uncertainty_experiment import UncertaintyExperiment
from src.experiments.visual_experiment import VisualExperiment
from src.experiments.aggr_metrics import *
from src.utils import YamlReader

dataset_configs = {
    'dfs' : 'configs/dfs/dataset.yaml',
    'prim' : 'configs/prim/dataset.yaml',
    'fractal' : 'configs/fractal/dataset.yaml',
}

model_configs = {
    'FcVAE' : 'configs/FcVAE_config.yaml',
    'ConvVAE' : 'configs/ConvVAE_config.yaml',
    'TransformerVAE' : 'configs/TransformerVAE_config.yaml',
}

trained_model_paths = {
    'dfs' : {
        'FcVAE' : 'saved_models/dfs/FcVAE.pt',
        'ConvVAE' : 'saved_models/dfs/ConvVAE.pt',
        'TransformerVAE' : 'saved_models/dfs/TransformerVAE.pt'
    },
    'prim' : {
        'FcVAE' : 'saved_models/prim/FcVAE.pt',
        'ConvVAE' : 'saved_models/prim/ConvVAE.pt',
        'TransformerVAE' : 'saved_models/prim/TransformerVAE.pt'
    },
    'fractal' : {
        'FcVAE' : 'saved_models/fractal/FcVAE.pt',
        'ConvVAE' : 'saved_models/fractal/ConvVAE.pt',
        'TransformerVAE' : 'saved_models/fractal/TransformerVAE.pt'
    }
}

aggr_metrics = {
    'aggr_branching_factor' : aggr_branching_factor,
    'aggr_connected_components' : aggr_connected_components,
    'aggr_count_holes_in_outer_wall' : aggr_count_holes_in_outer_wall,
    'aggr_has_path' : aggr_has_path,
    'aggr_keeps_shortest_path' : aggr_keeps_shortest_path,
    'aggr_ratio_straight_to_curl_paths' : aggr_ratio_straight_to_curl_paths
}

dataset_index = {
    'train' : 0,
    'val' : 1,
    'test' : 2
}

def style_experiment(args):
    """
    Method to run the style experiment (StyleExperiment class) with the given arguments.

    Args:
        args: argparse.Namespace; Arguments for the experiment
    """
    datasets = []
    model_tuples = []
    metrics = []
    # add the metrics to the list
    for metric in args.metrics:
        metrics.append(aggr_metrics[metric])
    # iterate over the datasets
    for d in args.datasets:
        # for each dataset, create a MazeDataset object
        dataset_config = dataset_configs[d]
        yr = YamlReader(dataset_config)
        oc = yr.read()
        datasets.append(yr.build_datasets(oc)[dataset_index[args.split]])
        models = []
        # Create a list of model instances for each dataset
        for model_name in args.models:
            yr.set_path(model_configs[model_name])
            model = yr.build_VAE(yr.read())
            model.load_state_dict(torch.load(trained_model_paths[d][model_name]))
            model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            models.append(model)
        # turn models list into a tuple
        models = tuple(models)
        model_tuples.append(models)
    # Run the experiment
    e = StyleExperiment(
        models=model_tuples,
        datasets=datasets,
        metrics=metrics,
        n=args.n,
        path=args.output_path
    )

    e.run()

def uncertainty_experiment(args):
    """
    Run the uncertainty experiment (UncertaintyExperiment class) with the given arguments.

    Args:
        args: argparse.Namespace; Arguments for the experiment
    """
    datasets = []
    model_tuples = []
    # iterate over the datasets
    for d in args.datasets:
        # for each dataset, create a MazeDataset object
        dataset_config = dataset_configs[d]
        yr = YamlReader(dataset_config)
        oc = yr.read()
        datasets.append(yr.build_datasets(oc)[dataset_index[args.split]])
        models = []
        # Create a list of model instances for each dataset
        for model_name in args.models:
            yr.set_path(model_configs[model_name])
            model = yr.build_VAE(yr.read())
            model.load_state_dict(torch.load(trained_model_paths[d][model_name]))
            model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            models.append(model)
        # turn models list into a tuple
        models = tuple(models)
        model_tuples.append(models)
    # Run the experiment
    e = UncertaintyExperiment(
        models=model_tuples,
        datasets=datasets,
        path="figures/"
    )
    e.run()

def visualizations(args):
    """
    Run the visualizations experiment (VisualExperiment class) with the given arguments.

    Args:
        args: argparse.Namespace; Arguments for the experiment
    """
    datasets = []
    model_tuples = []
    # iterate over the datasets
    for d in args.datasets:
        # for each dataset, create a MazeDataset object
        dataset_config = dataset_configs[d]
        yr = YamlReader(dataset_config)
        oc = yr.read()
        datasets.append(yr.build_datasets(oc)[dataset_index[args.split]])
        models = []
        # Create a list of model instances for each dataset
        for model_name in args.models:
            yr.set_path(model_configs[model_name])
            model = yr.build_VAE(yr.read())
            model.load_state_dict(torch.load(trained_model_paths[d][model_name]))
            model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            models.append(model)
        # turn models list into a tuple
        models = tuple(models)
        model_tuples.append(models)
    # Run the experiment
    e = VisualExperiment(
        models=model_tuples,
        datasets=datasets,
        path="figures/",
        n=args.n
    )
    e.run()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', '-e', type=str, required=True, choices=['style', 'visualizations', 'uncertainty'])
    parser.add_argument('--datasets', '-d', type=str, required=False, choices=dataset_configs.keys(), nargs='+',
                        default=list(dataset_configs.keys()))
    parser.add_argument('--models', '-m', type=str, required=False, nargs='+',
                        choices=['FcVAE', 'ConvVAE','TransformerVAE'],
                        default=['FcVAE', 'ConvVAE', 'TransformerVAE'])
    parser.add_argument('--metrics', '-met', type=str, required=False, nargs='+',
                        choices=['aggr_branching_factor', 'aggr_connected_components', 'aggr_count_holes_in_outer_wall',
                                 'aggr_has_path', 'aggr_keeps_shortest_path', 'aggr_ratio_straight_to_curl_paths'],
                        default=['aggr_branching_factor', 'aggr_connected_components', 'aggr_count_holes_in_outer_wall',
                                 'aggr_has_path', 'aggr_keeps_shortest_path', 'aggr_ratio_straight_to_curl_paths'])
    parser.add_argument('--split','-s', type=str, required=False, choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--output-path', '-o', type=str, required=False, default='results/maze_style_experiment.csv')
    parser.add_argument('--n', '-n', type=int, required=False, default=100)
    args = parser.parse_args()
    # Run the experiment based on the experiment type
    match args.experiment:
        case 'style':
            print(f'Running different style experiment on {args.datasets} datasets with {args.models} models and {len(args.metrics)} metrics.')
            style_experiment(args)
                    
        case 'visualizations':
            print(f'Running visualizations on {args.datasets} datasets with {args.models} models.')
            visualizations(args)
            
        case 'uncertainty':
            print(f'Running uncertainty experiment on {args.datasets} datasets with {args.models} models.')
            uncertainty_experiment(args)
        case _:
            print('Invalid experiment')


if __name__ == '__main__':
    main()