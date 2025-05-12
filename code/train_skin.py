
import torch
import os
import sys
import pathlib
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
sys.path.append(ROOT/"yolov9")

from models.yolo import Model 
from models.yolo import ClassificationModel
from utils.dataloaders import create_classification_dataloader
from utils.torch_utils import smart_optimizer, ModelEMA, reshape_classifier_output
from utils.general import TQDM_BAR_FORMAT, LOGGER, increment_path
from classify import val as validate
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda import amp
from copy import deepcopy
from tqdm import tqdm
import datetime
from multiprocessing import freeze_support
from torch.amp import GradScaler
from torch import nn
import torch.nn.functional as F
from utils.loggers import GenericLogger
import mlflow
import argparse




path = ROOT/"yolov9/weights/gelan-c.pt"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov5s-cls.pt', help='initial weights path')
    parser.add_argument('--data', type=str, default='imagenette160', help='cifar10, cifar100, mnist, imagenet, ...')
    parser.add_argument('--epochs', type=int, default=10, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=224, help='train, val image size (pixels)')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train-cls', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--pretrained', nargs='?', const=True, default=True, help='start from i.e. --pretrained False')
    parser.add_argument('--optimizer', choices=['SGD', 'Adam', 'AdamW', 'RMSProp'], default='Adam', help='optimizer')
    parser.add_argument('--lr0', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--decay', type=float, default=5e-5, help='weight decay')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing epsilon')
    parser.add_argument('--cutoff', type=int, default=None, help='Model layer cutoff index for Classify() head')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout (fraction)')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    parser.add_argument('--cat', choices=["skin", "other"], default="skin", help='name of dataset will be trained')
    return parser.parse_known_args()[0] if known else parser.parse_args()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: shape [batch, num_classes], raw output from model
        # targets: shape [batch], class indices
        ce_loss = F.cross_entropy(logits, targets, reduction='none').float()
        pt = torch.exp(-ce_loss)  # pt = softmax prob of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean().float()
        elif self.reduction == 'sum':
            return focal_loss.sum().float()
        return focal_loss.float()

def train():
    if mlflow.active_run():
        mlflow.end_run()
    opt = parse_opt()
    mlflow.start_run()
    mlflow.log_artifact("requirements.txt")
    model = torch.load(path, weights_only=False)
    datacat = opt.cat
    LOCAL_RANK = os.getenv('LOCAL_RANK', -1)
    RANK = -1
    if datacat == "skin":
        data_dir = pathlib.Path(f"../dataset_{datacat}")
        imgsz = (224, 224)
        epochs = 200
    else:
        data_dir = pathlib.Path(f"../dataset_{datacat}/phase1")
        imgsz = (380, 224)
        epochs = 65
    augment = True
    device = "cuda"
    bs = 16
    nw = 4
    
    cuda=True

    
    trainloader = create_classification_dataloader(
                            path=data_dir / 'train',
                            imgsz=imgsz,
                            batch_size=bs,
                            augment=augment,
                            rank=LOCAL_RANK,
                            workers=nw)
    
    
    opt.epochs = epochs
    opt.batch_size = bs
    opt.dropout = 0.3
    mlflow.log_param("epochs", opt.epochs)
    mlflow.log_param("batch_size", opt.batch_size)
    mlflow.log_param("imgsz", opt.imgsz)
    mlflow.log_param("optimizer", opt.optimizer)
    mlflow.log_param("lr0", opt.lr0)
    mlflow.log_param("decay", opt.decay)
    mlflow.log_param("label_smoothing", opt.label_smoothing)
    mlflow.log_param("dropout", opt.dropout)
    mlflow.log_param("data_augment", augment)
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    testloader = create_classification_dataloader(
                            path=data_dir / 'test',
                            imgsz=imgsz,
                            batch_size=bs,
                            augment=False,
                            rank=LOCAL_RANK,
                            workers=nw)
    
    best_fitness = 0.0
    clf = ClassificationModel(model=model["model"].float(), nc=2, cutoff=10, dropout_rate=opt.dropout, c_=256)
    reshape_classifier_output(clf, n=2)
    optimizer = smart_optimizer(clf, "Adam")
    lrf = 0.01  # final lr (fraction of lr0)
    lf = lambda x: (1 - x / epochs) * (1 - lrf) + lrf  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    ema = ModelEMA(clf) if RANK in {-1, 0} else None
    ema.ema.to(device)
    criterion = FocalLoss(gamma=2)
    logger = GenericLogger(opt, console_logger=LOGGER) if RANK in {-1, 0} else None
    scaler = GradScaler()
    wdir = pathlib.Path("../weights")
    last, best = wdir / 'last.pt', wdir / 'best.pt'
  
    for m in clf.modules():
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()

    for p in clf.parameters():
        p.requires_grad = True  # for training
    clf.to(device)

    for epoch in range(epochs):  # loop over the dataset multiple times
        tloss, vloss, fitness = 0.0, 0.0, 0.0  # train loss, val loss, fitness
        clf.train()
        if RANK != -1:
            trainloader.sampler.set_epoch(epoch)
        pbar = enumerate(trainloader)
        if RANK in {-1, 0}:
            pbar = tqdm(enumerate(trainloader), total=len(trainloader), bar_format=TQDM_BAR_FORMAT)
        for i, (images, labels) in pbar:  # progress bar
            images, labels = images.to(device, non_blocking=True), labels.to(device)

            # Forward
            with torch.amp.autocast(device_type="cuda"):  # stability issues when enabled
                loss = criterion(clf(images), labels)

            # Backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Optimize
            
            torch.nn.utils.clip_grad_norm_(clf.parameters(), max_norm=10.0)  # clip gradients

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema:
                ema.update(clf)

            if RANK in {-1, 0}:
                # Print
                tloss = (tloss * i + loss.item()) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                pbar.desc = f"{f'{epoch + 1}/{epochs}':>10}{mem:>10}{tloss:>12.3g}" + ' ' * 36
                
                # Test
                if i == len(pbar) - 1:  # last batch
                    top1, top5, vloss = validate.run(model=ema.ema,
                                                    dataloader=testloader,
                                                    imgsz=imgsz,
                                                    criterion=criterion,
                                                    pbar=pbar)  # test accuracy, loss
                    fitness = top1  # define fitness as top1 accuracy
                    mlflow.log_metric("train_loss", tloss, step=epoch)
                    mlflow.log_metric("test_loss", vloss, step=epoch)
                    mlflow.log_metric("top1_accuracy", top1, step=epoch)
                    mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)

            # Scheduler
        scheduler.step()

        # Log metrics
        if RANK in {-1, 0}:
            # Best fitness
            if fitness > best_fitness:
                best_fitness = fitness

            metrics = {
                "train/loss": tloss,
                "test/loss": vloss,
                "metrics/accuracy_top1": top1,
                "metrics/accuracy_top5": top5,
                "lr/0": optimizer.param_groups[0]['lr']}  # learning rate
            logger.log_metrics(metrics, epoch)
    
            # Save model
            final_epoch = epoch + 1 == epochs
            if  True or final_epoch:
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(ema.ema),  # deepcopy(de_parallel(model)).half(),
                    'ema': None,  # deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': None,  # optimizer.state_dict(),
                    'date': datetime.datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fitness:
                    torch.save(ckpt, best)
                    mlflow.log_artifact(str(best))

                    mlflow.pytorch.log_model(ema.ema, "model")   
                    del ckpt
    mlflow.end_run()


if __name__ == '__main__':
    freeze_support()
    
    train()

    
