import argparse

from src.utils import YamlReader

from src.Trainer import Trainer

import torch

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a VAE on a dataset.")
    parser.add_argument("--config-path", type=str, default="configs/ConvVAE_config.yaml", help="Path to the configuration file.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read configuration file
    yr = YamlReader(args.config_path)
    oc = yr.read()

    model = yr.build_VAE(oc)
    model = model.to(device)

    train_dataset, validation_dataset, test_dataset = yr.build_datasets(oc)

    training_params = yr.read_training_params(oc, model) # dictionary with keys "batch_size", "optimizer", "n_epochs"

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        batch_size=training_params['batch_size'],
        optimizer=training_params['optimizer'],
        loss_fn = None
    )

    trainer.train_loop(n_epochs=training_params['num_epochs'], step=trainer.auto_encode_step, model_name=training_params['model_name'])

if __name__ == "__main__":
    main()