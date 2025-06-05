"""Training stuffs"""

import os
import sys
import time
from tqdm.auto import tqdm
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from transformers import PreTrainedModel
from accelerate import Accelerator

# Ensure the parent directory is in the path for module imports
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))  # Add parent directory to path

from src.utils.misc_utils import N, LOG
from src.utils.train_utils import accuracy, my_f1_score, AverageMeter, EarlyStopping


def test(model: PreTrainedModel, accelerator: Accelerator, loader):
    """
    Evaluate model on test/validation set
    Loader can be either test_loader or val_loader
    """
    losses, accs, f1s = AverageMeter(), AverageMeter(), AverageMeter()
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()  # scalar value
    with torch.no_grad():
        for batch in loader:
            # Get data
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            y = batch["labels"]
            # Forward pass
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            with accelerator.autocast():
                loss = criterion(output.logits, y)
            # Metrics
            losses.update(N(loss), input_ids.size(0))
            acc = accuracy(N(output.logits), N(y))
            f1 = my_f1_score(N(output.logits), N(y))
            accs.update(acc, input_ids.size(0))
            f1s.update(f1, input_ids.size(0))

    return accs.avg, f1s.avg


def train_loop(opts, model, optimizer, scheduler, accelerator: Accelerator, train_loader, val_loader):
    """Training loop for training a pretrained model with given finetuning setting"""
    early_stopping = EarlyStopping(opts)
    start_epoch, step = 1, 0
    start_time = time.time()
    for epoch in range(start_epoch, opts.num_epochs + 1):

        step, val_acc = train_epoch(
            opts, model, optimizer, scheduler, accelerator, train_loader,
            val_loader, epoch, step)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            early_stopping(val_acc, model)
            if early_stopping.early_stop:
                LOG.info("Early stopping triggered, breaking training...")
                LOG.info("val_acc=%.3f <> best_val_acc=%.3f",
                         val_acc, early_stopping.best_score)
                break

    accelerator.end_training()
    runtime = time.time() - start_time
    if accelerator.is_main_process:
        LOG.info("Training completed with runtime=%.2fs, "
                 "ended at epoch=%d, step=%d", runtime, epoch, step)


def train_epoch(opts, model: PreTrainedModel, optimizer: Optimizer, scheduler: LRScheduler, accelerator: Accelerator, train_loader, val_loader, epoch, step):
    """Train for a single epoch"""
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(
        opts.class_weights, dtype=torch.float32, device=accelerator.device))  # scalar value
    losses, accs, f1s = AverageMeter(), AverageMeter(), AverageMeter()

    with tqdm(train_loader, unit="batch", disable=not accelerator.is_local_main_process) as tepoch:
        for batch_idx, batch in enumerate(tepoch):
            model.train()
            tepoch.set_description(f"{epoch:03d}")

            # Gradient accumulation
            with accelerator.accumulate(model):
                # Get data
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                y = batch["labels"]

                # Forward pass
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                with accelerator.autocast():
                    loss = criterion(output.logits, y)
                # Backward pass
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()  # at each training step

            # if batch_idx % opts.accum_steps == 0:
                # Metrics
                losses.update(N(loss), input_ids.size(0))
                acc = accuracy(N(output.logits), N(y))
                f1 = my_f1_score(N(output.logits), N(y))
                accs.update(acc, input_ids.size(0))
                f1s.update(f1, input_ids.size(0))

            if batch_idx % opts.log_every == 0 or batch_idx == len(train_loader) - 1:
                accelerator.wait_for_everyone()
                # Compute training metrics and log to comet_ml
                train_loss, train_acc, train_f1 = losses.avg, accs.avg, f1s.avg
                # Compute validation metrics and log to comet_ml
                val_acc, val_f1 = test(model, accelerator, val_loader)
                # Log to console
                tepoch.set_postfix(train_loss=train_loss, train_acc=train_acc, train_f1=train_f1,
                                   val_acc=val_acc, val_f1=val_f1)
                tepoch.update()
                step += 1

    return step, val_acc
