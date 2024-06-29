#!/bin/bash
python main.py --config-path configs/cond_test/cFcVAE_config.yaml
python main.py --config-path configs/cond_test/FcVAE_config.yaml
python main.py --config-path configs/cond_test/cConvVAE_config.yaml
python main.py --config-path configs/cond_test/cTransformerVAE_config.yaml

python main.py --config-path configs/cond_test/ConvVAE_config.yaml
python main.py --config-path configs/cond_test/TransformerVAE_config.yaml