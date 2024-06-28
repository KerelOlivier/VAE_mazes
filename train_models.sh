#!/bin/bash
python main.py --config-path configs/dfs/ConvVAE_config.yaml
python main.py --config-path configs/dfs/FcVAE_config.yaml
python main.py --config-path configs/dfs/TransformerVAE_config.yaml

python main.py --config-path configs/prim/ConvVAE_config.yaml
python main.py --config-path configs/prim/FcVAE_config.yaml
python main.py --config-path configs/prim/TransformerVAE_config.yaml

python main.py --config-path configs/fractal/ConvVAE_config.yaml
python main.py --config-path configs/fractal/FcVAE_config.yaml
python main.py --config-path configs/fractal/TransformerVAE_config.yaml
