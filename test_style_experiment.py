from src.experiments.maze_style_experiment import StyleExperiment
from src.utils import YamlReader
from src.MazeDataset import MazeDataset

import torch

if __name__ == "__main__":
    yr = YamlReader("configs/FcVAE_config.yaml")
    oc = yr.read()
    model = yr.build_VAE(oc)
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(torch.load('saved_models/FcVAE.pt'))
    dataset = yr.build_datasets(oc)[0]
    e = StyleExperiment(
        models = [
            model
        ],
        datasets=[dataset],
        metrics=[]
    )
    e.run()