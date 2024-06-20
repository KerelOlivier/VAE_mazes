from torch.utils.data import Dataset
import numpy as np

class MazeDataset(Dataset):
    def __init__(self, file_path, idx_range, transform=None):
        """
        Dataset class for storing/retrieving mazes and path (solutions) stored in npy format.

        Args:
            file_path: str; path to the npy file
            idx_range: tuple; range of indices to use
            transform: function; transformation to apply to the data (optional)
        """
        self.data = np.load(file_path)
        self.idx_range = idx_range
        self.data = self.data[idx_range[0]:idx_range[1], :, :, :]
        self.width = self.data.shape[2]
        self.height = self.data.shape[3]
        self.transform = transform

    def __len__(self):
        return self.idx_range[1] - self.idx_range[0]

    def __getitem__(self, idx):
        """
        Get the maze and path (solution) at the specified index.

        Args:
            idx: int; index of the maze & path to retrieve

        Returns:
            maze: np.ndarray; maze with shape (1, W, H)
            path: np.ndarray; path with shape (1, W, H)
        """
        if self.transform:
            x,y = self.transform(self.data[idx, 0, :, :]), self.transform(self.data[idx, 1, :, :])
        else:
            x,y = self.data[idx, 0, :, :].reshape(1, self.width, self.height), self.data[idx, 1, :, :].reshape(1, self.width, self.height)
        return x, y
    
