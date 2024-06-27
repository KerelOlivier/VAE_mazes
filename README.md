# VAE maze generation
This project aims to generate mazes using a Conditional Variational Auto Encoder. Why? Because why not?

## Setup
https://beckham.nz/2023/04/27/conditional-vaes.html


## Running instructions
To train a model, use the `main.py` access point (in the root directory).

It supports the following arguments:

    -h, --help            show this help message and exit
    --config-path CONFIG_PATH
                        Path to the configuration file.

To change parameters for the model or training, update the YAML config file (at --config-path). configs/README.md contains the documentation for how to update the YAML configs.

An example run: \
```python main.py --config-path configs/fractal/FcVAE_config.yaml``` \
This will train the fully connected VAE on the FractalTessellation dataset, using the configuration specified at ```configs/fractal/FcVAE_config.yaml```

To run an experiment, use the `run_experiments.py` access point (in the root directory).

It supports the following arguments:

    -h, --help            show this help message and exit
    --experiment {style,visualizations,uncertainty}, -e {style,visualizations,uncertainty}
        The experiment to run, pick one of the above
    --datasets [{dfs,prim,fractal} ...], -d [{dfs,prim,fractal} ...]
        The datasets to use, specify a subset of the above
    --models, -m [{FcVAE,ConvVAE,TransformerVAE} ...]
        The models to use, specify a subset of the above
    --metrics, -met [{aggr_branching_factor,aggr_connected_components,aggr_count_holes_in_outer_wall,aggr_has_path,aggr_keeps_shortest_path,aggr_ratio_straight_to_curl_paths} ...]
        The metrics to use, specify a subset from the above
    --split {train,val,test}, -s {train,val,test}
        The dataset to use (train/val/test)
    --output-path 
        The output file path for the csv files (with style experiment)
    --n, -n
        Style experiment:
            The number of samples to compute statistics over (default=100)
        Visualizations/Uncertainty experiment:
            The number of samples/reconstructions to generate and visualize (default=4)

Example run: \
`python run_experiments.py -e style -d fractal -m FcVAE -met aggr_ratio_straight_to_curl_paths -n 100` \
To run the style experiment, with only the fractal FcVAE model, reporting only the ratio straight to curled paths, computed over 100 samples

More complex example run: \
`python run_experiments.py -e visualizations -d fractal prim dfs -m FcVAE TransformerVAE -n 4 `
Run the visualizations experiment, on all datasets, the fully connected and attention model, and produce 4 samples to plot.
## Results


