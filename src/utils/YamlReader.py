from omegaconf import OmegaConf

from src.MazeDataset import MazeDataset
from src.models.StandardNormalPrior import StandardNormalPrior
from src.models.MogPrior import MogPrior
from src.models.VAE import VAE
from src.models.FcVAE import FcEncoder, FcDecoder
from src.models.TransformerVAE import TransformerEncoder, TransformerDecoder
from src.models.ConvVAE import ConvEncoder, ConvDecoder
import torch
from torchvision.transforms import ToTensor, Compose


def partial(func, *args, **keywords):
    """
    Return a new partial object which when called will behave like func called with the positional arguments args and keyword arguments keywords.
    """
    def newfunc(*fargs, **fkeywords):
        newkeywords = keywords.copy()
        newkeywords.update(fkeywords)
        return func(*(args + fargs), **newkeywords)
    newfunc.func = func
    newfunc.args = args
    newfunc.keywords = keywords
    return newfunc

class YamlReader:
    # TODO: Update when new classes are added
    DATASET_CLASSES = {"MazeDataset": MazeDataset}
    MODEL_CLASSES = {"VAE": VAE}
    PRIOR_CLASSES = {"StandardNormalPrior": StandardNormalPrior, "MogPrior": MogPrior}
    ENCODER_CLASSES = {"FcEncoder": FcEncoder, "TransformerEncoder": TransformerEncoder, "ConvEncoder": ConvEncoder}
    DECODER_CLASSES = {"FcDecoder": FcDecoder, "TransformerDecoder": TransformerDecoder, "ConvDecoder": ConvDecoder}
    OPTIMIZER_DICT = {"Adam": torch.optim.Adam}
    DATASET_TRANSFORMS = {"ToTensor": torch.FloatTensor, "Flatten": torch.flatten, "Normalize": torch.nn.functional.normalize, "Compose": torch.nn.Sequential,
                          "Unsqueeze": partial(torch.unsqueeze, dim=0)}

    def __init__(self, path):
        self.path = path

    def set_path(self, path):
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
            decoder=decoder_class(**oc["model"]["decoder_params"]),
            is_conditional=oc["model"]["is_conditional"],
            name=oc["model"]["name"]
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

        
        transform = [YamlReader.DATASET_TRANSFORMS[tf] for tf in oc["dataset"]["transforms"]["sequential"]]
        transform = Compose(transform)
        name = oc["dataset"]["name"]
        dataset_class = YamlReader.DATASET_CLASSES[oc["dataset"]["class"]]
        train_dataset = dataset_class(file_path=file_path, **oc["dataset"]["train_params"], transform=transform, name="train "+name)
        validation_dataset = dataset_class(file_path=file_path, **oc["dataset"]["validation_params"], transform=transform, name="validation "+name)
        test_dataset = dataset_class(file_path=file_path, **oc["dataset"]["test_params"], transform=transform, name="test "+name)

        return train_dataset, validation_dataset, test_dataset
    
    def read_training_params(self, oc, model):
        """
        Read training parameters from the OmegaConf object.

        Args:
            oc: OmegaConf; OmegaConf object with the training configuration
            model: nn.Module; model to train
            
        Returns:
            dict; dictionary with keys "batch_size", "optimizer"

        """
        opt = YamlReader.OPTIMIZER_DICT[oc["training"]["optimizer"]]
        opt = opt(model.parameters(), **oc["training"]["optimizer_params"])
        annealer_params = oc["training"]["annealer_params"]
        return {"batch_size": oc["training"]["batch_size"], "optimizer": opt, "num_epochs": oc["training"]["num_epochs"], "model_name": oc["training"]["model_name"], "annealer_params": annealer_params}