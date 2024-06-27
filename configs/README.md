# YAML system

There are two types of configs here: only dataset configs, and training model configs.

## Only dataset
Holds (nested) dictionaries, data and dataset. \
data specifies the data file location (in file_path).

    data:
        file_path: data/<file_path>.npy

dataset contains information on the instantiation of the pytorch Dataset implementation. 

    dataset:
        class: MazeDataset - the Dataset class to use
        name: <name> - name of the dataset
        transforms:
            sequential: [<t1>, <t2>] - the torchvision transforms to use in the MazeDataset
        <split>_params:
            idx_range: [<start_idx>, <end_idx>] - defines the splits to use for the <split> dataset. For example [0, 100] would use samples 0 - 99 for this dataset.

## Training config

The training configs contain the above dictionaries data and dataset. And additionally define two more; model and training

In model, define the parameterization of the model.

    model:
        class: VAE - use the VAE template class
        encoder_class: <class> - class to use as encoder
        decoder_class: <class> - class to use as decoder
        is_conditional: <bool> - whether to use conditional VAE
        name: <name> - model name
        prior_class: <class> - class to use for the prior
        <component>_params: prior/encoder/decoder params
            <param>: <val>
    
    training:
        model_name: <name>.pt - where to store your model file
        batch_size: <bs> - batch size
        num_epochs: <e> - number of epochs
        optimizer: Adam
        optimizer_params:
            lr: <lr> - learning rate
            weight_decay: <wd> - weight decay