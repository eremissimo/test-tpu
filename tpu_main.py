import os
from argparse import ArgumentParser
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tpu_models import soft_iou_loss, hausdorff_loss, \
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
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["l2reg"])
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["lr_gamma"])
    val_metrics = mtr.MetricCollection({"tacc": mtr.Accuracy(compute_on_step=False, ignore_index=0),
                                        "iou": IoU(num_classes=N_CLASSES, ignore_index=0, reduction='none',
                                                   compute_on_step=False)},
                                       prefix="Valid/").to(device)
    class_weights = torch.tensor([0.0028, 2.2711, 0.5229, 1.2033], device=device)   # precomputed from entire dataset
    haus_weight = config["hausdorff_loss_weight"]

    xm.master_print("Training ........ ")
    train_iou_loss = torch.tensor(0., dtype=torch.float32, device=device)
    val_iou_loss = torch.tensor(0., dtype=torch.float32, device=device)
    train_haus_loss = torch.tensor(0., dtype=torch.float32, device=device)
    val_haus_loss = torch.tensor(0., dtype=torch.float32, device=device)

    # 5. TRAINING & VALIDATION LOOPS
    for epoch in range(config["epochs"]):
        train_loader_with_tqdm = tqdm(train_device_loader, total=train_num_batches, desc=f"Epoch {epoch}",
                                      disable=not xm.is_master_ordinal())
        # 5.1 training loop
        model.train()
        train_iou_loss.zero_()
        train_haus_loss.zero_()
        for img, seg_t, dist_t in train_loader_with_tqdm:
            optimizer.zero_grad()
            logits = model(img)
            haus_loss = hausdorff_loss(logits, seg_t, dist_t, weight=class_weights)
            iou_loss = soft_iou_loss(logits, seg_t)
            (iou_loss + haus_weight*haus_loss).backward()
            xm.optimizer_step(optimizer)
            train_iou_loss += iou_loss.detach()
            train_haus_loss += haus_loss.detach()
        lr_scheduler.step()

        # 5.2 validation loop
        model.eval()
        val_iou_loss.zero_()
        val_haus_loss.zero_()
        with torch.no_grad():
            for img, seg_t, dist_t in val_device_loader:
                logits = model(img)
                iou_loss_v = soft_iou_loss(logits, seg_t)
                haus_loss_v = hausdorff_loss(logits, seg_t, dist_t, weight=class_weights)
                val_iou_loss += iou_loss_v
                val_haus_loss += haus_loss_v
                val_metrics.update(logits, seg_t)

        # 5.3 training and validation metrics
        metrics: dict = val_metrics.compute()
        val_metrics.reset()
        metrics.update({
            "Train/iouloss": train_iou_loss/train_num_batches,
            "Train/hausloss": train_haus_loss/train_num_batches,
            "Valid/iouloss": val_iou_loss/val_num_batches,
            "Valid/hausloss": val_haus_loss/val_num_batches
        })
        metrics_reduced = reduce_dict("metrics_reduce", metrics)
        train_metrics_reduced = {k: v for k, v in metrics_reduced.items() if k.startswith("Train")}
        valid_metrics_reduced = {k: v for k, v in metrics_reduced.items() if k.startswith("Valid")}
        xm.master_print(f" Epoch {epoch} training: {train_metrics_reduced} \n",
                        f"Epoch {epoch} validation: {valid_metrics_reduced}")

    # xm.master_print(met.metrics_report())
    xm.rendezvous("done!")


def reduce_fn(x):
    return sum(x)/len(x)


def reduce_val(tag: str, x: torch.Tensor):
    x_reduced = xm.mesh_reduce(tag, x, reduce_fn)
    return x_reduced


def reduce_dict(tag: str, x: dict):
    """ Concat all values to a single tensor, reduce it across all tpu cores, then reconstruct the original dict """
    cat_tensor = torch.hstack(tuple(x.values()))
    sizes = [val.numel() for val in x.values()]
    cat_tensor_reduced = xm.mesh_reduce(tag, cat_tensor, reduce_fn)
    split_cat_tensor = (val.squeeze() for val in torch.split(cat_tensor_reduced, sizes, dim=0))
    x.update(zip(x, split_cat_tensor))
    return x


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
    parser.add_argument("--hausdorff_loss_weight", type=float, default=10,
                        help=" weight of hausdorff loss (default: 10) ")

    args = parser.parse_args()
    config = args.__dict__

    # WRAPPED_MODEL = xmp.MpModelWrapper(SImple(n_chan=config["base_channels"], use_norm=config["use_batchnorm"]))
    model = Conv232RefineNet(n_chan=config["base_channels"], spatial_size=128,
                             use_norm=config["use_batchnorm"], leaping_dim=2)
    WRAPPED_MODEL = xmp.MpModelWrapper(model)
    SERIAL_EXEC = xmp.MpSerialExecutor()

    xmp.spawn(map_fn, args=(config,), nprocs=8, start_method='fork')
