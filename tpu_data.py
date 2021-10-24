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
        t_dt = torch.randn((n_train, 4, *spatial_shape))
        v_dt = torch.randn((n_val, 4, *spatial_shape))
        train_data_tensors = (t_img, t_seg, t_dt)
        val_data_tensors = (v_img, v_seg, v_dt)
        print("Beware: all replicas have different generated data tensors! ")
    else:
        train_path = os.path.join(data_path, "train.pt")
        train_dt_path = os.path.join(data_path, "train_dt.pt")
        val_path = os.path.join(data_path, "test.pt")
        val_dt_path = os.path.join(data_path, "test_dt.pt")
        os.makedirs(data_path, exist_ok=True)
        if os.path.isfile(train_path):
            print("train.pt is found in the data folder. Skipping download phase ... ")
        else:
            load_from_bucket(config["bucket"], train_path)
            load_from_bucket(config["bucket"], train_dt_path)
        if os.path.isfile(val_path):
            print("test.pt is found in the data folder. Skipping download phase ... ")
        else:
            load_from_bucket(config["bucket"], val_path)
            load_from_bucket(config["bucket"], val_dt_path)
        train_img, train_seg = torch.load(train_path)
        train_dt: torch.Tensor = torch.load(train_dt_path)                # segm target border distance transform
        val_img, val_seg = torch.load(val_path)
        val_dt: torch.Tensor = torch.load(val_dt_path)
        train_data_tensors = tuple(ff.pad(x, [0, 0, 4, 4, 4, 4]) for x in [train_img, train_seg, train_dt])
        val_data_tensors = tuple(ff.pad(x, [0, 0, 4, 4, 4, 4]) for x in [val_img, val_seg, val_dt])
        gc.collect()

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


def save_to_bucket(file_path, bucket_name):
    """ Upload a file to the root of the specified bucket """
    storage_client = storage.Client()
    maybe_bucket = storage_client.lookup_bucket(bucket_name=bucket_name)
    if maybe_bucket is None:
        maybe_bucket = storage_client.create_bucket(bucket_name, user_project="airy-coil-321017",
                                                    location="europe-west4-a")
    filename = os.path.basename(file_path)
    blob = maybe_bucket.blob(filename)
    blob.upload_from_filename(file_path)
    print(f"Saved {filename} to gs://{bucket_name} successfully ~")


def random_x_flip(image: torch.Tensor, segm: torch.Tensor, dt: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ The only used transform here is a random flip along the X axis """
    if torch.rand((1,)) > 0.5:
        image = image.flip(2)   # x axis  [B, C, X, Y, Z]
        segm = segm.flip(1)     # x axis  [B, X, Y, Z]
        dt = dt.flip(2)         # x axis  [B, C, X, Y, Z]
    return image, segm, dt


def random_x_flip_tuple(img_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return random_x_flip(*img_tuple)


def to_float32_int64(img_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    img, seg, dt = img_tuple
    return img.float(), seg.long(), dt.float()


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