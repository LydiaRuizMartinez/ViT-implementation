# deep learning libraries
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# other libraries
import os
import requests
import tarfile
import shutil
from requests.models import Response
from tarfile import TarFile
from PIL import Image


class ImagenetteDataset(Dataset):
    """
    This class is the Imagenette Dataset.
    """

    def __init__(self, path: str) -> None:
        """
        Constructor of ImagenetteDataset.

        Args:
            path: path of the dataset.
        """

        # set attributes
        self.path = path
        self.names = os.listdir(path)

    def __len__(self) -> int:
        """
        This method returns the length of the dataset.

        Returns:
            length of dataset.
        """

        return len(self.names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """
        This method loads an item based on the index.

        Args:
            index: index of the element in the dataset.

        Returns:
            tuple with image and label. Image dimensions:
                [channels, height, width].
        """

        # load image path and label
        image_path: str = f"{self.path}/{self.names[index]}"
        label: int = int(self.names[index].split("_")[0])

        # load image
        transformations = transforms.Compose([transforms.ToTensor()])
        image = Image.open(image_path)
        image = transformations(image)

        return image, label


def load_data(
    path: str, batch_size: int = 128, num_workers: int = 0
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    This function returns two Dataloaders, one for train data and
    other for validation data for imagenette dataset.

    Args:
        path: path of the dataset.
        color_space: color_space for loading the images.
        batch_size: batch size for dataloaders. Default value: 128.and
        num_workers: number of workers for loading data.
            Default value: 0.

    Returns:
        tuple of dataloaders, train, val and test in respective order.
    """

    # download folders if they are not present
    if not os.path.isdir(f"{path}"):
        # create main dir
        os.makedirs(f"{path}")

        # download data
        download_data(path)

    # create datasets
    train_dataset: Dataset = ImagenetteDataset(f"{path}/train")
    val_dataset: Dataset
    train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2])
    test_dataset: Dataset = ImagenetteDataset(f"{path}/val")

    # define dataloaders
    train_dataloader: DataLoader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataloader: DataLoader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_dataloader: DataLoader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return train_dataloader, val_dataloader, test_dataloader


def download_data(path: str) -> None:
    """
    This function downloads the data from internet.

    Args:
        path: path to dave the data.
    """

    # define paths
    url: str = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
    target_path: str = f"{path}/imagenette2.tgz"

    # download tar file
    response: Response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, "wb") as f:
            f.write(response.raw.read())

    # extract tar file
    tar_file: TarFile = tarfile.open(target_path)
    tar_file.extractall(path)
    tar_file.close()

    # create final save directories
    os.makedirs(f"{path}/train")
    os.makedirs(f"{path}/val")

    # define resize transformation
    transform = transforms.Resize((64, 64))

    # loop for saving processed data
    list_splits: tuple[str, str] = ("train", "val")
    for i in range(len(list_splits)):
        list_class_dirs = os.listdir(f"{path}/imagenette2/{list_splits[i]}")
        for j in range(len(list_class_dirs)):
            list_dirs = os.listdir(
                f"{path}/imagenette2/{list_splits[i]}/{list_class_dirs[j]}"
            )
            for k in range(len(list_dirs)):
                image = Image.open(
                    f"{path}/imagenette2/{list_splits[i]}/"
                    f"{list_class_dirs[j]}/{list_dirs[k]}"
                )
                image = transform(image)
                if image.im.bands == 3:
                    image.save(f"{path}/{list_splits[i]}/{j}_{k}.jpg")

    # delete other files
    os.remove(target_path)
    shutil.rmtree(f"{path}/imagenette2")

    return None
