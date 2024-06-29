#!/bin/bash
echo "Training TransformerVAE"
python main.py --config-path configs/cond_test/TransformerVAE_config.yaml
echo "Training cTransformerVAE"
python main.py --config-path configs/cond_test/cTransformerVAE_config.yaml
echo "Training ConvVAE"
python main.py --config-path configs/cond_test/ConvVAE_config.yaml
echo "Training cConvVAE"
python main.py --config-path configs/cond_test/cConvVAE_config.yaml
echo "Training FcVAE"
python main.py --config-path configs/cond_test/FcVAE_config.yaml
echo "Training cFcVAE"
python main.py --config-path configs/cond_test/cFcVAE_config.yaml