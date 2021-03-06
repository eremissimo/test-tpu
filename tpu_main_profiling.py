import os
from argparse import ArgumentParser
import torch
import torch.nn.functional as ff
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tpu_models import SImple
from tpu_data import download_datasets
from tqdm import tqdm
import torchmetrics as mtr

os.environ["XLA_USE_BF16"] = '1'
os.environ["XLA_HLO_DEBUG"] = '1'

import torch_xla
# import torch_xla.debug.metrics as met
import torch_xla.debug.profiler as xp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp


DATA_PATH = os.path.join(os.getcwd(), "data")
N_CLASSES = 4


def map_fn(index: int, config) -> None:
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

    # 3. MODELS & METRICS
    model = WRAPPED_MODEL.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["lr_gamma"])
    val_metrics = mtr.MetricCollection({"acc": mtr.Accuracy(compute_on_step=False),
                                        "tacc": mtr.Accuracy(compute_on_step=False, ignore_index=0)},
                                       prefix="Valid/").to(device)
    val_loss_accumulator = mtr.AverageMeter(compute_on_step=False).to(device)
    class_weights = torch.tensor([0.0028, 2.2711, 0.5229, 1.2033], device=device)   # precomputed from entire dataset

    server = xp.start_server(9012)      # profiler

    xm.master_print("Training ........ ")

    # 5. TRAINING & VALIDATION LOOPS
    train_avg_loss = torch.tensor(0., device=device)
    for epoch in range(config["epochs"]):
        train_loader_with_tqdm = tqdm(train_device_loader, total=train_num_batches, desc=f"Epoch {epoch}",
                                      disable=not xm.is_master_ordinal())
        # 5.1 training loop
        model.train()
        train_avg_loss.zero_()
        for img, seg_t in train_loader_with_tqdm:
            with xp.StepTrace('train_step'):
                with xp.Trace('forward-backward'):
                    optimizer.zero_grad()
                    logits = model(img)
                    loss = ff.cross_entropy(logits, seg_t, weight=class_weights)
                    loss.backward()
                xm.optimizer_step(optimizer)
                lr_scheduler.step()
                train_avg_loss += loss.detach()


        # 5.2 validation loop
        model.eval()
        with torch.no_grad():
            for img, seg_t in val_device_loader:
                with xp.StepTrace('test_step'):
                    logits = model(img)
                    loss_v = ff.cross_entropy(logits, seg_t, weight=class_weights)
                    val_loss_accumulator.update(loss_v)
                    val_metrics.update(logits, seg_t)
        # per-device compute metrics and then all_reduce them across devices
        train_avg_loss_reduced = reduce_val("train_loss_reduce", train_avg_loss/train_num_batches)
        val_loss_reduced = reduce_val("val_loss_reduce", val_loss_accumulator.compute())
        val_metrics_reduced = reduce_dict("val_metrics_reduce", val_metrics.compute())

        val_metrics.reset()
        val_loss_accumulator.reset()
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


def reduce_val_and_dict(tag: str, value: torch.Tensor, dictionary: dict):
    value_reduced = xm.mesh_reduce(tag + "_value", value, reduce_fn)
    dict_reduced = reduce_dict(tag+"_dict", dictionary)
    return value_reduced, dict_reduced


def get_ce_weights(seg_t: torch.Tensor) -> torch.Tensor:
    """ Normalization coefficients for CE loss """
    counts = torch.flatten(seg_t).bincount(minlength=N_CLASSES)
    weights = 1.0/counts
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

    WRAPPED_MODEL = xmp.MpModelWrapper(SImple(n_chan=config["base_channels"], use_norm=config["use_batchnorm"]))
    # WRAPPED_MODEL = xmp.MpModelWrapper(SImpleDirac(n_chan=config["base_channels"]))
    SERIAL_EXEC = xmp.MpSerialExecutor()

    xmp.spawn(map_fn, args=(config,), nprocs=8, start_method='fork')
