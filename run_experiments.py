import argparse
import torch

from src.experiments.maze_style_experiment import StyleExperiment
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
    'aggr_keeps_shortest_path' : aggr_keeps_shortest_path
}

dataset_index = {
    'train' : 0,
    'val' : 1,
    'test' : 2
}

def style_experiment(args):
    datasets = []
    model_tuples = []
    metrics = []

    for metric in args.metrics:
        metrics.append(aggr_metrics[metric])

    for d in args.datasets:
        dataset_config = dataset_configs[d]
        yr = YamlReader(dataset_config)
        oc = yr.read()
        datasets.append(yr.build_datasets(oc)[dataset_index[args.split]])
        models = []
        for model_name in args.models:
            yr.set_path(model_configs[model_name])
            model = yr.build_VAE(yr.read())
            model.load_state_dict(torch.load(trained_model_paths[d][model_name]))
            model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            models.append(model)
        # turn models list into a tuple
        models = tuple(models)
        model_tuples.append(models)
        
    e = StyleExperiment(
        models=model_tuples,
        datasets=datasets,
        metrics=metrics,
        n=100,
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
                                 'aggr_has_path', 'aggr_keeps_shortest_path'],
                        default=['aggr_branching_factor', 'aggr_connected_components', 'aggr_count_holes_in_outer_wall',
                                 'aggr_has_path', 'aggr_keeps_shortest_path'])
    parser.add_argument('--split','-s', type=str, required=False, choices=['train', 'val', 'test'], default='test')
    args = parser.parse_args()

    match args.experiment:
        case 'style':
            style_experiment(args)
                    
            print('Running style transfer experiment')
        case 'visualizations':
            print('Running visualizations experiment')
        case 'uncertainty':
            print('Running uncertainty experiment')
        case _:
            print('Invalid experiment')

    for dataset in args.datasets:
        print(f'Running experiment on {dataset} dataset')

    for model in args.models:
        print(f'Running experiment with {model} model')
if __name__ == '__main__':
    main()