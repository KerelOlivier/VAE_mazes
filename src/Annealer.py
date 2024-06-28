import numpy as np

class Annealer:
    """
    Inspired by https://github.com/hubertrybka/vae-annealing/tree/main
    """
    def __init__(self, total_epochs, annealing_type, beta_0, cyclical=False, disable=False):
        """
        Annealer class for annealing the VAE NELBO KL term.

        Args:
            total_epochs: int; total number of epochs to train for
            annealing_type: str; type of annealing to use, options are linear, exponential, and cosine
            beta_0: float; initial value of beta
            cyclical: bool; whether to use cyclical annealing or not
            disable: bool; whether to disable annealing or not
        """
        self.total_epochs = total_epochs
        self.annealing_type = annealing_type
        self.beta_0 = beta_0
        self.cyclical = cyclical
        if disable:
            self.beta_0 = 1.0
            self.annealing_type = "none"

    def get_beta(self, current_epoch):
        """
        Get the current value of beta based on the annealing type.

        Returns:
            float; current value of beta
        """
        if self.cyclical:
            current_epoch = current_epoch % self.total_epochs

        if self.annealing_type == "linear":
            beta = current_epoch / self.total_epochs
        elif self.annealing_type == "cosine":
            beta = (np.cos(np.pi * (current_epoch / self.total_epochs - 1)) + 1) / 2
        elif self.annealing_type == "exponential":
            beta = 1 - np.exp(-5 * current_epoch / self.total_epochs)
        elif self.annealing_type == "logistic":
            exponent = self.total_epochs / 2 - current_epoch
            beta = 1 / (1 + np.exp(exponent))
        elif self.annealing_type == "none":
            beta = 1.0
        else:
            raise ValueError("Invalid annealing type")

        beta = self.beta_0 + beta * (1 - self.beta_0)
        return beta
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Make 4 subplots, one for each annealing type
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    annealing_types = ["linear", "cosine", "exponential", "logistic"]
    for i, annealing_type in enumerate(annealing_types):
        annealer = Annealer(total_epochs=100, annealing_type=annealing_type, beta_0=0.0, cyclical=False, disable=False)
        betas = [annealer.get_beta(epoch) for epoch in range(100)]
        ax = axs[i // 2, i % 2]
        ax.plot(betas)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Beta")
        ax.set_title(annealing_type.capitalize())
    plt.tight_layout()
    plt.show()

    annealer = Annealer(total_epochs=25, annealing_type="cosine", beta_0=0.0, cyclical=True, disable=False)
    betas = [annealer.get_beta(epoch) for epoch in range(100)]
    
    plt.plot(betas)
    plt.xlabel("Epoch")
    plt.ylabel("Beta")
    plt.title("Cyclical Cosine Annealing")
    plt.show()