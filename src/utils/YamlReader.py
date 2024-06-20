from omegaconf import OmegaConf

from src.MazeDataset import MazeDataset
from src.models import *


class YamlReader:
    # TODO: Update when new classes are added
    DATASET_CLASSES = {"MazeDataset": MazeDataset}
    MODEL_CLASSES = {"VAE": VAE}
    PRIOR_CLASSES = {}
    ENCODER_CLASSES = {}
    DECODER_CLASSES = {}

    def __init__(self, path):
        self.path = path

    def read(self):
        return OmegaConf.load(self.path)
    
    def build_VAE(self, oc):
        """
        Build a VAE model from the OmegaConf object.

        Args:
            oc: OmegaConf; OmegaConf object with the VAE configuration

        Returns:
            VAE; VAE model
        """
        vae_class = YamlReader.MODEL_CLASSES[oc["model"]["class"]]
        prior_class = YamlReader.PRIOR_CLASSES[oc["model"]["prior_class"]]
        encoder_class = YamlReader.ENCODER_CLASSES[oc["model"]["encoder_class"]]
        decoder_class = YamlReader.DECODER_CLASSES[oc["model"]["decoder_class"]]


        return vae_class(
            prior=prior_class(**oc["model"]["prior_params"]),
            encoder=encoder_class(**oc["model"]["encoder_params"]),
            decoder=decoder_class(**oc["model"]["decoder_params"])
        )
    
    def build_datasets(self, oc):
        """
        Build datasets from the OmegaConf object.

        Args:
            oc: OmegaConf; OmegaConf object with the dataset configuration

        Returns:
            Dataset; training dataset
            Dataset; validation dataset
            Dataset; test dataset
        """
        file_path = oc["data"]["file_path"]

        dataset_class = YamlReader.DATASET_CLASSES[oc["dataset"]["class"]]
        train_dataset = dataset_class(file_path=file_path, **oc["dataset"]["train_params"])
        validation_dataset = dataset_class(file_path=file_path, **oc["dataset"]["validation_params"])
        test_dataset = dataset_class(file_path=file_path, **oc["dataset"]["test_params"])

        raise train_dataset, validation_dataset, test_dataset