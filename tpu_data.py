import os
import gc
import torch
import torch.nn.functional as ff
from torch.utils.data import Dataset, TensorDataset
from torchvision.transforms import Compose
from typing import Tuple, Optional, Callable
from google.cloud import storage


def download_datasets(config: dict, data_path: str) -> Tuple[Dataset, Dataset]:
    if config["synthetic_data"] or not config["bucket"]:
        print("Using synthetic data... ")
        n_train = 20
        n_val = 5
        spatial_shape = (64, 64, 32)
        t_img = torch.randn((n_train, 4, *spatial_shape))
        v_img = torch.randn((n_val, 4, *spatial_shape))
        t_seg = torch.randint(low=0, high=3, size=(n_train, *spatial_shape))
        v_seg = torch.randint(low=0, high=3, size=(n_val, *spatial_shape))
        train_data_tensors = (t_img, t_seg)
        val_data_tensors = (v_img, v_seg)
    else:
        train_path = os.path.join(data_path, "train.pt")
        val_path = os.path.join(data_path, "test.pt")
        os.makedirs(data_path, exist_ok=True)
        if os.path.isfile(train_path):
            print("train.pt is found in data folder. Skipping download phase ... ")
        else:
            load_from_bucket(config["bucket"], train_path)
        if os.path.isfile(val_path):
            print("test.pt is found in data folder. Skipping download phase ... ")
        else:
            load_from_bucket(config["bucket"], val_path)
        train_data_tensors: Tuple[torch.Tensor, torch.Tensor] = torch.load(train_path)  # img, target_segmentation
        val_data_tensors: Tuple[torch.Tensor, torch.Tensor] = torch.load(val_path)
        defect = train_data_tensors[0].shape[2] % 128
        if defect != 0:
            print(f"Extra padding: {train_data_tensors[0].shape} -> ", end="")
            defect = (128 - defect) // 2
            pad_dim = [0, 0] + [defect]*4
            train_data_tensors = ff.pad(train_data_tensors[0], pad_dim), ff.pad(train_data_tensors[1], pad_dim)
            val_data_tensors = ff.pad(val_data_tensors[0], pad_dim), ff.pad(val_data_tensors[1], pad_dim)
            gc.collect()
            print(train_data_tensors[0].shape)
    print("Instantiating dataset classes...  ", end="")

    transform = Compose([random_x_flip_tuple, to_float32_int64])
    train_dataset = TransformTensorDataset(train_data_tensors, transform=transform)
    val_dataset = TransformTensorDataset(val_data_tensors, transform=transform)
    print("OK!")
    return train_dataset, val_dataset


def load_from_bucket(bucket_name: str, file_path: str):
    """ Load a tensor .pt file from the root of a bucket to a local file_path """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    filename = os.path.basename(file_path)
    blob = bucket.blob(filename)
    blob.download_to_filename(file_path)
    print(f"Downloaded {filename} from gs://{bucket_name} successfully ~")


def random_x_flip(image: torch.Tensor, segm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ The only used transform here is a random flip along the X axis """
    if torch.rand((1,)) > 0.5:
        image = image.flip(2)   # x axis  [B, C, X, Y, Z]
        segm = segm.flip(1)     # x axis  [B, X, Y, Z]
    return image, segm


def random_x_flip_tuple(img_tuple: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    return random_x_flip(img_tuple[0], img_tuple[1])


def to_float32_int64(img_tuple: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    return img_tuple[0].float(), img_tuple[1].long()


class TransformTensorDataset(TensorDataset):
    """ TensorDataset with transforms enabled """
    def __init__(self, tensors: Tuple[torch.Tensor, ...], transform: Optional[Callable] = None):
        super().__init__(*tensors)
        self.transform = transform

    def __getitem__(self, item) -> Tuple[torch.Tensor, ...]:
        elem = TensorDataset.__getitem__(self, item)
        if self.transform is not None:
            elem = self.transform(elem)
        return elem