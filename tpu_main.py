import os
import itertools
from argparse import ArgumentParser
import torch
import torch.nn.functional as ff
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tpu_models import focal_loss, recall_ce_loss, soft_iou_loss, \
    SImple, Conv232Unet, Conv232RefineNet, Conv232RefineNetCascade
from tpu_data import download_datasets
from tqdm import tqdm
import torchmetrics as mtr
from torchmetrics.classification import IoU


import torch_xla
# import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp


DATA_PATH = os.path.join(os.getcwd(), "data")
N_CLASSES = 4


def map_fn(index: int, config: dict) -> None:
    torch.manual_seed(111)
    device = xm.xla_device()
    
    # 1. DATASETS
    brats_train_dataset, brats_val_dataset = SERIAL_EXEC.run(lambda: download_datasets(config, data_path=DATA_PATH))

    # 2. DATALOADERS
    train_sampler = DistributedSampler(brats_train_dataset, num_replicas=xm.xrt_world_size(),
                                       rank=xm.get_ordinal(), shuffle=True)
    val_sampler = DistributedSampler(brats_val_dataset, num_replicas=xm.xrt_world_size(),
                                     rank=xm.get_ordinal(), shuffle=False)
    train_loader = DataLoader(brats_train_dataset, batch_size=config["batch_size"], shuffle=False,
                              sampler=train_sampler, num_workers=config["dataworkers"])
    val_loader = DataLoader(brats_val_dataset, batch_size=config["batch_size"], shuffle=False,
                            sampler=val_sampler, num_workers=config["dataworkers"])
    train_device_loader = pl.MpDeviceLoader(train_loader, device)
    val_device_loader = pl.MpDeviceLoader(val_loader, device)
    train_num_batches = len(train_device_loader)
    val_num_batches = len(val_device_loader)

    # 3. MODELS & METRICS
    model = WRAPPED_MODEL.to(device)
    model.train()
    per_parameter_optimizer_options = [{"params": model.contr_adjust.parameters(), "weight_decay": 0.0},
                                       {"params": itertools.chain.from_iterable(sub.parameters() for sub in
                                                                               model.children() if sub is not
                                                                               model.contr_adjust),
                                        "weight_decay": config["l2reg"]}]
    optimizer = optim.Adam(per_parameter_optimizer_options, lr=config["lr"])
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["lr_gamma"])
    val_metrics = mtr.MetricCollection({"acc": mtr.Accuracy(compute_on_step=False),
                                        "tacc": mtr.Accuracy(compute_on_step=False, ignore_index=0),
                                        "iou": IoU(num_classes=N_CLASSES, ignore_index=0, reduction='none',
                                                   compute_on_step=False)},
                                       prefix="Valid/").to(device)
    class_weights = torch.tensor([0.0028, 2.2711, 0.5229, 1.2033], device=device)   # precomputed from entire dataset


    xm.master_print("Training ........ ")
    train_avg_loss = torch.tensor(0., device=device)
    val_avg_loss = torch.tensor(0., device=device)

    # 5. TRAINING & VALIDATION LOOPS
    for epoch in range(config["epochs"]):
        train_loader_with_tqdm = tqdm(train_device_loader, total=train_num_batches, desc=f"Epoch {epoch}",
                                      disable=not xm.is_master_ordinal())
        # 5.1 training loop
        model.train()
        train_avg_loss.zero_()
        for img, seg_t in train_loader_with_tqdm:
            optimizer.zero_grad()
            logits = model(img)
            loss = recall_ce_loss(logits, seg_t)
            loss.backward()
            xm.optimizer_step(optimizer)
            train_avg_loss += loss.detach()
        lr_scheduler.step()

        # 5.2 validation loop
        model.eval()
        val_avg_loss.zero_()
        with torch.no_grad():
            for img, seg_t in val_device_loader:
                logits = model(img)
                loss_v = recall_ce_loss(logits, seg_t)
                val_avg_loss += loss_v
                val_metrics.update(logits, seg_t)

        # 5.3 training and validation metrics
        train_avg_loss_reduced = reduce_val("train_loss_reduce", train_avg_loss/train_num_batches)
        val_loss_reduced = reduce_val("val_loss_reduce", val_avg_loss/val_num_batches)
        val_metrics_reduced = reduce_dict("val_metrics_reduce", val_metrics.compute())
        val_metrics.reset()
        xm.master_print(f" Epoch {epoch} training: curr loss = {loss},  avg loss = {train_avg_loss_reduced} \n",
                        f"Epoch {epoch} validation: avg loss = {val_loss_reduced},  {val_metrics_reduced}")

    # xm.master_print(met.metrics_report())


def reduce_fn(x):
    return sum(x)/len(x)


def reduce_val(tag: str, x: torch.Tensor):
    x_reduced = xm.mesh_reduce(tag, x, reduce_fn)
    return x_reduced


def reduce_dict(tag: str, x: dict):
    x_reduced = {k: xm.mesh_reduce(tag + k, v, reduce_fn) for k, v in x.items()}
    return x_reduced


def get_ce_weights(seg_t: torch.Tensor) -> torch.Tensor:
    """ Normalization coefficients for CE loss """
    # torch.bincount is not supported by xla unfortunately
    counts = torch.stack([(seg_t == i).sum() for i in range(N_CLASSES)]).float()
    weights = torch.where(counts > 0., 1.0/counts, torch.zeros((1,), device=counts.device))
    weights = N_CLASSES*weights/weights.sum()
    return weights


if __name__ == "__main__":
    parser = ArgumentParser(description="Hyperparameters")
    parser.add_argument("--epochs", type=int, default=10,
                        help="training epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate (default: 1e-3)")
    parser.add_argument("--l2reg", type=float, default=0.0,
                        help="L2 regularization of neural network weights (default: 0.0)")
    parser.add_argument("--lr_gamma", type=float, default=0.98,
                        help="gamma parameter for exponential lr_scheduler (default: 0.98)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="batch size for dataloaders (default: 1)")
    parser.add_argument("--dataworkers", type=int, default=0,
                        help="number of workers for dataloders (default: 0)")
    parser.add_argument("--bucket", type=str, default='',
                        help="a bucket where fullram dataset tensors are stored (default:'')")
    parser.add_argument("--base_channels", type=int, default=8,
                        help=" nnet base channels (default: 8) ")
    parser.add_argument("--use_batchnorm", action="store_true",
                        help=" use batchnorm in the model ")
    parser.add_argument("--synthetic_data", action="store_true",
                        help=" use synthetic random data instead of downloading examples from a bucket ")

    args = parser.parse_args()
    config = args.__dict__

    # WRAPPED_MODEL = xmp.MpModelWrapper(SImple(n_chan=config["base_channels"], use_norm=config["use_batchnorm"]))
    model = Conv232RefineNet(n_chan=config["base_channels"], spatial_size=128,
                             use_norm=config["use_batchnorm"], leaping_dim=2)
    WRAPPED_MODEL = xmp.MpModelWrapper(model)
    SERIAL_EXEC = xmp.MpSerialExecutor()

    xmp.spawn(map_fn, args=(config,), nprocs=8, start_method='fork')
