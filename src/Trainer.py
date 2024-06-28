import torch
from torch.utils.data import DataLoader
from src.models import VAE
from src.Annealer import Annealer
from src.EarlyStopper import EarlyStopper

import matplotlib.pyplot as plt

class Trainer:
    def __init__(
        self, model, train_dataset, validation_dataset, batch_size, optimizer, loss_fn
    ) -> None:
        """
        Trainer class for training models on datasets.

        Args:
            model: nn.Module; model to train
            train_dataset: Dataset; training dataset
            validation_dataset: Dataset; validation dataset
            batch_size: int; batch size for training
            optimizer: torch.optim; optimizer for training
            loss_fn: torch.nn; loss function to use
        """
        # Set model, datasets, batch size, optimizer, loss function, and device
        self.model = model
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Initialize loaders
        self.train_loader = None
        self.validation_loader = None

        # Set loaders
        self.make_loaders()

    def make_loaders(self):
        """
        Make loaders for training and validation datasets.
        """
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        self.validation_loader = DataLoader(
            self.validation_dataset, batch_size=self.batch_size, shuffle=False
        )

    def train_loop(self, n_epochs, step, save_model=True, model_name="unnamed_model.pt",
                   annealing_type="none", beta_0=1.0, cyclical=False, disable=False,
                   patience=5, min_delta=0.05):
        """
        Train the model for n_epochs using the specified step function.

        Args:
            n_epochs: int; number of epochs to train for
            step: function; step function to use for training, options are auto_encode_step
            save_model: bool; whether to save the model or not
            model_name: str; name of the model to save
        """
        assert self.train_loader != None and self.validation_loader != None
        early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
        annealer = Annealer(total_epochs=n_epochs, annealing_type=annealing_type, beta_0=beta_0, cyclical=cyclical, disable=disable)
        losses = []
        vlosses = []
        # Train for n_epochs, store best validation loss and save model if save_model is True
        best_vloss = 1_000_000
        for epoch in range(1, n_epochs + 1):
            self.beta = annealer.get_beta(epoch)
            # Use the specified step function to train the model
            avg_loss, avg_vloss = step()
            losses.append(avg_loss)
            vlosses.append(avg_vloss)

            print(
                f"Epoch {epoch} -  Training loss: {avg_loss}, Validation loss: {avg_vloss}"
            )
            if save_model and avg_vloss < best_vloss:
                best_vloss = avg_vloss
                torch.save(self.model.state_dict(), "saved_models/" + model_name)
                
            if early_stopper.early_stop(avg_vloss):
                print("Early stopping...")
                break
        
        plt.plot(losses, label="Training Loss")
        plt.plot(vlosses, label="Validation Loss")
        plt.legend()
        plt.savefig(f"figures/{model_name[:-3]}_loss.png")

    def auto_encode_step(self):
        """
        Predict the next state using an AR 1 step dataset.

        Returns:
            avg_loss: float; average training loss
            avg_vloss: float; average validation loss
            avg_loss_dict: dict; dictionary of average training losses
            avg_vloss_dict: dict; dictionary of average validation losses
        """
        # Training
        self.model.train(True)
        running_loss = 0
        avg_loss = 0
        
        for i, data in enumerate(self.train_loader):
            # Every object in train_loader is the current state x and a future state y
            x, y = data
            x, y = x.to(self.device), y.to(self.device)
            # Set gradients to zero for new batch
            self.optimizer.zero_grad()

            if self.model.is_conditional:
                # For conditional models, pass x and y to the model
                loss = self.model(x=x, y=y, beta=self.beta)
            else:
                # For non-conditional models, pass x to the model
                loss = self.model(x=x, beta=self.beta)

            loss.backward()

            # Clip the gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Adjust weights of the model
            self.optimizer.step()

            running_loss += loss.item()
            
        total_items = len(self.train_dataset)

        # Report training metrics
        avg_loss = running_loss / (total_items)

        # Validation
        self.model.eval()
        running_vloss = 0
        avg_vloss = 0

        with torch.no_grad():
            for i, vdata in enumerate(self.validation_loader):
                # Every object in validation_loader is the current state x and a future state y
                val_x, val_y = vdata
                
                val_x, val_y = val_x.to(self.device), val_y.to(self.device)

                if self.model.is_conditional:
                    # For conditional models, pass x and y to the model
                    vloss = self.model(x=val_x, y=val_y, beta=self.beta)
                else:
                    # For non-conditional models, pass x to the model
                    vloss = self.model(x=val_x, beta=self.beta)

                running_vloss += vloss.item()
                
        total_items = len(self.validation_dataset)

        avg_vloss = running_vloss / (total_items)

        return avg_loss, avg_vloss